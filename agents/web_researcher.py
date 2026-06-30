"""Web researcher agent — searches the web and summarises results for each sub-question.

All Tavily searches run in parallel via ThreadPoolExecutor. All summarisation is
done in a single batched LLM call (one call for all questions), cutting LLM
round-trips from 3 to 1 and avoiding Groq rate-limit retries.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from pydantic import BaseModel, Field
from tavily import TavilyClient

from graph.state import ResearchState, WebResult
from llm.client import call_structured, call_with_backoff
from llm.tracing import create_span

logger = logging.getLogger(__name__)

_MAX_RESULTS_PER_Q = 3
_MAX_SNIPPET_CHARS = 200


# ---------------------------------------------------------------------------
# LLM response schema — one summary per sub-question in a single call
# ---------------------------------------------------------------------------


class QuestionSummary(BaseModel):
    """Summary for one sub-question."""

    summary: str = Field(description="One-paragraph summary of the search results.")


class WebSummaryBatch(BaseModel):
    """Batch response: one entry per sub-question, in the same order."""

    summaries: list[QuestionSummary] = Field(
        description="One summary per input sub-question, in order."
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a financial research assistant. You will receive a numbered list of research
questions, each with web search results. For each question write one concise factual
paragraph summarising the key findings. Do not invent data not in the results.

Respond with ONLY valid JSON:
{"summaries": [{"summary": "paragraph for Q1"}, {"summary": "paragraph for Q2"}, ...]}

Do not include any text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_results(client: TavilyClient, question: str) -> tuple[str, list[dict]]:
    """Run a Tavily search for one sub-question and return (question, raw_results).

    Args:
        client: Initialised TavilyClient.
        question: Research sub-question.

    Returns:
        Tuple of (question, list of result dicts from Tavily).
    """
    response = client.search(
        query=question,
        max_results=_MAX_RESULTS_PER_Q,
        include_answer=False,
        search_depth="basic",
    )
    return question, response.get("results", [])


def _summarise_all(
    questions_with_results: list[tuple[str, list[dict]]],
) -> list[str]:
    """Summarise all sub-questions in a single batched LLM call.

    Args:
        questions_with_results: List of (question, raw_results) tuples in order.

    Returns:
        List of summary strings, one per question (may be shorter on LLM failure).
    """
    blocks = []
    for i, (question, results) in enumerate(questions_with_results):
        snippets = "\n".join(
            f"  - {r.get('title','')}: {r.get('content','')[:_MAX_SNIPPET_CHARS]}"
            for r in results[:_MAX_RESULTS_PER_Q]
        )
        blocks.append(f"{i + 1}. Q: {question}\n{snippets or '  (no results)'}")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]
    try:
        batch: WebSummaryBatch = call_with_backoff(
            call_structured,
            messages=messages,
            response_schema=WebSummaryBatch,
        )
        return [s.summary for s in batch.summaries]
    except Exception as exc:
        logger.warning("Batch web summarisation failed: %s. Using empty summaries.", exc)
        return ["" for _ in questions_with_results]


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


def web_researcher_node(state: ResearchState) -> ResearchState:
    """Search the web for each sub-question and summarise all results in one LLM call.

    Tavily searches run in parallel (ThreadPoolExecutor). All summaries are
    generated in a single batched LLM call to minimise Groq API round-trips.
    Falls back to an empty list on any failure.

    Args:
        state: Current pipeline state. Reads ``sub_questions`` and ``job_id``.

    Returns:
        Updated state with ``web_results`` populated (may be empty on failure).
    """
    job_id = state.get("job_id", "unknown")
    sub_questions = state.get("sub_questions", [])[:3]

    span = create_span(
        "web_researcher_node",
        trace_id=state.get("langfuse_trace_id"),
        input_data={"sub_questions": sub_questions, "job_id": job_id},
    )

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_key:
        warning = "TAVILY_API_KEY not set — web_researcher returning empty results."
        logger.warning(warning)
        span.update(level="WARNING", status_message=warning, output={"web_results": []})
        span.end()
        return {"web_results": []}  # type: ignore[return-value]

    client = TavilyClient(api_key=tavily_key)

    # Step 1: Fetch all questions in parallel (no LLM involved)
    ordered_questions = list(sub_questions)
    raw_map: dict[str, list[dict]] = {q: [] for q in ordered_questions}

    with ThreadPoolExecutor(max_workers=len(ordered_questions) or 1) as pool:
        future_to_q = {pool.submit(_fetch_results, client, q): q for q in ordered_questions}
        for future in as_completed(future_to_q):
            try:
                question, results = future.result()
                raw_map[question] = results
            except Exception as exc:
                logger.warning("Tavily fetch failed: %s. Skipping.", exc)

    questions_with_results = [(q, raw_map[q]) for q in ordered_questions]

    # Step 2: Summarise all questions in ONE LLM call
    summaries = _summarise_all(questions_with_results)

    # Step 3: Build WebResult objects
    now = datetime.now(UTC)
    all_results: list[WebResult] = []
    for i, (question, results) in enumerate(questions_with_results):
        summary = summaries[i] if i < len(summaries) else ""
        for r in results:
            all_results.append(
                WebResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("content", "")[:_MAX_SNIPPET_CHARS],
                    summary=summary,
                    retrieved_at=now,
                )
            )

    span.update(output={"result_count": len(all_results)})
    span.end()
    logger.info("web_researcher_node collected %d results for job %s.", len(all_results), job_id)
    return {"web_results": all_results}  # type: ignore[return-value]
