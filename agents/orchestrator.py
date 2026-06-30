"""Orchestrator agent — decomposes the user query into sub-questions and a research plan.

This is the first node in the pipeline. Its output feeds all four parallel worker agents.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from graph.state import ResearchState
from llm.client import call_structured, call_with_backoff
from llm.tracing import create_span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class OrchestratorResponse(BaseModel):
    """Structured output expected from the orchestrator LLM call."""

    sub_questions: list[str] = Field(
        description="3–5 specific research sub-questions derived from the user query."
    )
    research_plan: str = Field(
        description="A concise paragraph describing the research approach and priorities."
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert equity research orchestrator. Given a user query about a company or sector,
your job is to decompose it into 3–5 specific research sub-questions and produce a brief
research plan.

You MUST respond with ONLY valid JSON that matches this exact schema:
{
  "sub_questions": ["question 1", "question 2", ...],
  "research_plan": "A paragraph describing the research approach."
}

Do not include any text outside the JSON object. Do not wrap it in markdown code fences.
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


def orchestrator_node(state: ResearchState) -> ResearchState:
    """Decompose the user query into sub-questions and a research plan.

    Starts a LangFuse span, calls the LLM via call_with_backoff(call_structured, ...),
    and writes sub_questions and research_plan into the state. On any error, sets
    state["error"] and returns without crashing the pipeline.

    Args:
        state: Current pipeline state. Reads ``query`` and ``job_id``.

    Returns:
        Updated state with ``sub_questions`` and ``research_plan`` populated.
    """
    query = state.get("query", "")
    job_id = state.get("job_id", "unknown")

    span = create_span(
        "orchestrator_node",
        trace_id=state.get("langfuse_trace_id"),
        input_data={"query": query, "job_id": job_id},
    )

    try:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Research query: {query}"},
        ]

        response: OrchestratorResponse = call_with_backoff(
            call_structured,
            messages=messages,
            response_schema=OrchestratorResponse,
        )

        span.update(
            output={
                "sub_questions": response.sub_questions,
                "research_plan": response.research_plan,
            }
        )
        span.end()

        logger.info(
            "Orchestrator produced %d sub-questions for job %s.",
            len(response.sub_questions),
            job_id,
        )

        updated: ResearchState = {
            **state,  # type: ignore[misc]
            "sub_questions": response.sub_questions,
            "research_plan": response.research_plan,
        }
        return updated

    except Exception as exc:
        error_msg = f"orchestrator_node failed: {exc}"
        logger.error(error_msg)
        span.update(level="ERROR", status_message=error_msg)
        span.end()
        return {**state, "error": error_msg}  # type: ignore[return-value]
