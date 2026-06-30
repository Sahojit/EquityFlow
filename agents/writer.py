"""Writer agent — synthesises all research inputs into a structured ResearchNote.

On revision passes, the previous critique is injected into the prompt so the
LLM can address unsupported claims. All LLM calls go through
call_with_backoff(call_structured, ...).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from graph.state import (
    CritiqueItem,
    FinancialData,
    NewsResult,
    ResearchNote,
    ResearchState,
    WebResult,
)
from llm.client import call_structured, call_with_backoff
from llm.tracing import create_span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior equity research analyst at a top-tier investment bank.
Write a detailed, 4–6 page investment research note based on the provided data.

CRITICAL RULES:
1. Respond with ONLY valid JSON matching the ResearchNote schema below. No markdown fences, no prose outside JSON.
2. Cite EVERY factual claim in full_text with a URL from the provided sources list. Use inline markdown links [source](url).
3. For any financial figure where data_available is False, write "data unavailable" rather than inventing a number.
4. The analyst field must always be "AlphaAgents v1".
5. full_text must be a complete markdown-formatted research note with sections: Executive Summary, Investment Thesis, Financial Analysis, Key Risks, Valuation, Comparables, Recommendation.

ResearchNote JSON schema:
{
  "company_name": "string",
  "ticker": "string",
  "analyst": "AlphaAgents v1",
  "date_generated": "ISO 8601 UTC datetime string",
  "investment_thesis": "string (2-4 sentences)",
  "key_risks": ["risk 1", "risk 2", ...],
  "valuation_summary": "string",
  "comparable_companies": ["TICKER1", "TICKER2", ...],
  "recommendation": "Buy" | "Hold" | "Sell" | "Not Rated",
  "confidence": "High" | "Medium" | "Low",
  "citations": ["url1", "url2", ...],
  "full_text": "full markdown note string"
}
"""


_MAX_WEB_RESULTS = 4
_MAX_NEWS_RESULTS = 4
_MAX_SNIPPET_CHARS = 120
_MAX_CITATION_URLS = 8
_MAX_PLAN_CHARS = 200


def _build_user_message(
    query: str,
    research_plan: str,
    web_results: list[WebResult],
    financial_data: FinancialData | None,
    news_results: list[NewsResult],
    memory_context: str,
    critique: list[CritiqueItem],
) -> str:
    """Assemble a token-budgeted user message for the writer LLM call.

    Inputs are truncated so the full request (prompt + max_tokens response)
    stays within the Groq free-tier 6,000 TPM limit per request.

    Args:
        query: Original user research query.
        research_plan: Research plan from the orchestrator.
        web_results: List of web search results with summaries.
        financial_data: Financial metrics (may have data_available=False).
        news_results: Recent news articles with sentiment.
        memory_context: Past research context from ChromaDB (omitted to save tokens).
        critique: List of CritiqueItems from a previous critic pass (empty on first pass).

    Returns:
        A formatted string ready to send as the user message.
    """
    sections: list[str] = [
        f"## Research Query\n{query}",
        f"## Research Plan\n{research_plan[:_MAX_PLAN_CHARS]}",
    ]

    # Financial data — compact single block
    if financial_data:
        fd = financial_data
        if fd.data_available:
            fin_block = (
                f"Ticker: {fd.ticker} | Company: {fd.company_name} | "
                f"Price: {fd.current_price} | P/E: {fd.pe_ratio} | "
                f"P/B: {fd.pb_ratio} | ROE: {fd.roe} | "
                f"RevGrowth: {fd.revenue_growth_yoy} | D/E: {fd.debt_to_equity} | "
                f"MarketCap: {fd.market_cap}"
            )
        else:
            fin_block = (
                f"Ticker: {fd.ticker} | Company: {fd.company_name} | "
                "Financial data unavailable — mark all figures as 'data unavailable'."
            )
        sections.append(f"## Financial Data\n{fin_block}")

    # Web results — truncated snippets, capped count
    top_web = web_results[:_MAX_WEB_RESULTS]
    if top_web:
        web_block = "\n\n".join(
            f"[{r.title}]({r.url})\n{r.summary[:_MAX_SNIPPET_CHARS]}"
            for r in top_web
        )
        sections.append(f"## Web Research\n{web_block}")

    # News — summary only, capped count
    top_news = news_results[:_MAX_NEWS_RESULTS]
    if top_news:
        news_block = "\n".join(
            f"[{n.headline}]({n.url}) [{n.sentiment.upper()}] {n.summary[:_MAX_SNIPPET_CHARS]}"
            for n in top_news
        )
        sections.append(f"## Recent News\n{news_block}")

    # Critique — revision pass only, unresolved items only
    if critique:
        unresolved = [c for c in critique if c.verdict in ("unsupported", "missing_citation")]
        if unresolved:
            critique_block = "\n".join(
                f"- \"{c.claim}\" → {c.reason}"
                for c in unresolved
            )
            sections.append(
                f"## Fix These Claims\n{critique_block}\n"
                "Add a citation URL or remove/qualify each claim above."
            )

    # Citation URLs — capped to avoid bloat
    all_urls = [r.url for r in top_web] + [n.url for n in top_news]
    unique_urls = list(dict.fromkeys(all_urls))[:_MAX_CITATION_URLS]
    if unique_urls:
        sections.append(f"## Citation URLs\n" + "\n".join(f"- {u}" for u in unique_urls))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


def writer_node(state: ResearchState) -> ResearchState:
    """Synthesise all research data into a structured ResearchNote.

    On first pass, critique is empty. On revision passes, the critique from the
    previous critic run is injected into the prompt. Increments revision_count
    each time it is called (first call sets it to 0 if unset, then leaves
    incrementing to the pipeline router — actually, we increment here on revision).

    Args:
        state: Current pipeline state. Reads web_results, financial_data,
            news_results, memory_context, research_plan, query, critique.

    Returns:
        Updated state with ``draft_note`` populated and ``revision_count`` incremented.
    """
    query = state.get("query", "")
    job_id = state.get("job_id", "unknown")
    revision_count = state.get("revision_count", 0)

    span = create_span(
        "writer_node",
        trace_id=state.get("langfuse_trace_id"),
        input_data={
            "query": query,
            "job_id": job_id,
            "revision_count": revision_count,
            "has_critique": bool(state.get("critique")),
        },
    )

    try:
        user_message = _build_user_message(
            query=query,
            research_plan=state.get("research_plan", ""),
            web_results=state.get("web_results", []),
            financial_data=state.get("financial_data"),  # type: ignore[arg-type]
            news_results=state.get("news_results", []),
            memory_context=state.get("memory_context", ""),
            critique=state.get("critique", []),
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        draft_note: ResearchNote = call_with_backoff(
            call_structured,
            messages=messages,
            response_schema=ResearchNote,
            max_tokens=1500,
        )

        # Ensure date_generated is set (LLM may omit it or use wrong format)
        if not draft_note.date_generated:
            draft_note = draft_note.model_copy(
                update={"date_generated": datetime.now(UTC)}
            )

        span.update(
            output={
                "recommendation": draft_note.recommendation,
                "confidence": draft_note.confidence,
                "citation_count": len(draft_note.citations),
            }
        )
        span.end()

        logger.info(
            "writer_node produced draft note for %s (%s). Revision=%d. job=%s",
            draft_note.company_name,
            draft_note.recommendation,
            revision_count,
            job_id,
        )

        return {
            **state,  # type: ignore[misc]
            "draft_note": draft_note,
            "revision_count": revision_count + 1,
        }

    except Exception as exc:
        error_msg = f"writer_node failed: {exc}"
        logger.error(error_msg)
        span.update(level="ERROR", status_message=error_msg)
        span.end()
        return {**state, "error": error_msg}  # type: ignore[return-value]
