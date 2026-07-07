"""Web researcher agent — searches the web and summarises results for each sub-question."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from pydantic import BaseModel, Field
from tavily import TavilyClient

from graph.state import ResearchState, WebResult
from llm.client import call_structured, call_with_backoff, get_llm_client, trace_span

logger = logging.getLogger(__name__)

_MAX_SUB_QUESTIONS = 3
_MAX_RESULTS_PER_Q = 5


# ---------------------------------------------------------------------------
# LLM response schema
# ---------------------------------------------------------------------------


class SnippetSummary(BaseModel):
    """One-to-two sentence summary of a single search result snippet."""

    summary: str = Field(description="1-2 sentence summary of the snippet.")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a financial research assistant. You will receive a web search result snippet.
Summarise it in 1-2 concise sentences, focused on facts relevant to equity research.
Do not invent information not present in the snippet.

Respond with ONLY valid JSON:
{"summary": "1-2 sentence summary"}

Do not include any text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


def web_researcher_node(state: ResearchState) -> ResearchState:
    """Search the web for each of the first 3 sub-questions and summarise each result.

    For every sub-question, runs a Tavily search and, for each result, calls the LLM
    to summarise the snippet into a WebResult. If Tavily fails for a given
    sub-question, logs a warning and continues to the next one rather than crashing.

    Args:
        state: Current pipeline state. Reads ``sub_questions`` and ``job_id``.

    Returns:
        Updated state with ``web_results`` populated (may be empty on failure).
    """
    job_id = state.get("job_id", "unknown")
    sub_questions = state.get("sub_questions", [])[:_MAX_SUB_QUESTIONS]

    try:
        get_llm_client()  # validates GROQ_API_KEY is configured before proceeding
        tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        web_results: list[WebResult] = []

        for sub_q in sub_questions:
            try:
                response = tavily.search(query=sub_q, max_results=_MAX_RESULTS_PER_Q)
            except Exception as exc:
                logger.warning(
                    "Tavily search failed for sub-question %r: %s. Skipping.", sub_q, exc
                )
                continue

            for result in response.get("results", []):
                url = result.get("url", "")
                title = result.get("title", "")
                snippet = result.get("content", "")

                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": snippet},
                ]
                summary_response: SnippetSummary = call_with_backoff(
                    call_structured,
                    messages=messages,
                    response_schema=SnippetSummary,
                )

                web_results.append(
                    WebResult(
                        url=url,
                        title=title,
                        snippet=snippet,
                        summary=summary_response.summary,
                        retrieved_at=datetime.now(UTC),
                    )
                )

        trace_span(
            "web_researcher_node",
            input_data={"sub_questions": sub_questions, "job_id": job_id},
            output_data={"result_count": len(web_results)},
        )

        logger.info(
            "web_researcher_node collected %d results for job %s.", len(web_results), job_id
        )

        return {**state, "web_results": web_results}  # type: ignore[return-value]

    except Exception as exc:
        error_msg = f"web_researcher_node failed: {exc}"
        logger.error(error_msg)
        trace_span(
            "web_researcher_node",
            input_data={"sub_questions": sub_questions, "job_id": job_id},
            output_data={"error": error_msg},
        )
        existing_error = state.get("error")
        combined_error = f"{existing_error}; {error_msg}" if existing_error else error_msg
        return {**state, "error": combined_error}  # type: ignore[return-value]
