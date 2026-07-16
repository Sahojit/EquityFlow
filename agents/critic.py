"""Critic agent — reviews the draft note for unsupported claims.

Uses the LLM as a judge. Returns a list of CritiqueItems (always a list, even if empty).
If the LLM call fails, returns an empty list so the pipeline routes to HITL rather
than crashing.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from graph.state import CritiqueItem, ResearchNote, ResearchState
from llm.client import call_structured, call_with_backoff
from llm.tracing import create_span

logger = logging.getLogger(__name__)


class CritiqueResponse(BaseModel):
    """Wrapper schema for the critic LLM output."""

    critique: list[CritiqueItem] = Field(
        description="List of reviewed claims. Empty list if no issues found."
    )


_SYSTEM_PROMPT = """\
You are a rigorous equity research compliance reviewer. Your job is to review an
investment research note and identify claims that are:
  - "unsupported": stated as fact but not backed by any cited source in the note
  - "missing_citation": a factual claim that has no inline URL citation
  - "supported": clearly backed by a cited source

Review up to 10 of the most important factual claims in the note.

Respond with ONLY valid JSON matching this schema:
{
  "critique": [
    {
      "claim": "verbatim claim text",
      "verdict": "supported" | "unsupported" | "missing_citation",
      "reason": "brief explanation"
    }
  ]
}

If the note is well-cited and all claims are supported, return {"critique": []}.
Do not include any text outside the JSON object.
"""


def critic_node(state: ResearchState) -> ResearchState:
    """Review the draft note and populate the critique list.

    If the LLM call fails, logs an error and returns an empty critique list
    so the revision_router routes to HITL rather than looping infinitely.

    Args:
        state: Current pipeline state. Reads ``draft_note`` and ``job_id``.

    Returns:
        Updated state with ``critique`` populated.
    """
    job_id = state.get("job_id", "unknown")
    draft_note: ResearchNote | None = state.get("draft_note")

    logger.info("Critic reviewing note for job %s", job_id)

    span = create_span(
        "critic_node",
        trace_id=state.get("langfuse_trace_id"),
        input_data={"job_id": job_id, "has_draft": draft_note is not None},
    )

    if draft_note is None:
        warning = "critic_node: no draft_note in state. Returning empty critique."
        logger.warning(warning)
        span.update(level="WARNING", status_message=warning, output={"critique": []})
        span.end()
        return {**state, "critique": []}

    try:
        note_excerpt = draft_note.full_text[:6000]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Research note to review:\n\n{note_excerpt}\n\n"
                    f"Known citations in the note: {draft_note.citations}"
                ),
            },
        ]

        response: CritiqueResponse = call_with_backoff(
            call_structured,
            messages=messages,
            response_schema=CritiqueResponse,
        )

        critique = response.critique

        unresolved_count = sum(
            1 for c in critique if c.verdict in ("unsupported", "missing_citation")
        )

        span.update(
            output={
                "total_claims": len(critique),
                "unresolved": unresolved_count,
            }
        )
        span.end()

        logger.info("Found %d critique items for job %s.", len(critique), job_id)
        if unresolved_count:
            logger.warning("Unsupported claims: %d (job=%s)", unresolved_count, job_id)

        return {**state, "critique": critique}

    except Exception as exc:
        error_msg = f"critic_node failed: {exc}. Returning empty critique."
        logger.error(error_msg)
        span.update(level="ERROR", status_message=error_msg)
        span.end()
        return {**state, "critique": []}
