"""Financial data agent — extracts a ticker from the query and fetches metrics via yfinance.

If yfinance returns no data, sets data_available=False so the writer can flag
uncertain figures explicitly rather than hallucinating them.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import yfinance as yf
from pydantic import BaseModel, Field

from graph.state import FinancialData, ResearchState
from llm.client import call_structured, call_with_backoff
from llm.tracing import create_span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM response schema for ticker extraction
# ---------------------------------------------------------------------------


class TickerExtraction(BaseModel):
    """LLM output when extracting a ticker symbol from a natural-language query."""

    ticker: str = Field(
        description="The stock ticker symbol (e.g. 'INFY', 'AAPL'). "
        "Use 'UNKNOWN' if you cannot determine it."
    )
    company_name: str = Field(
        description="Full legal company name (e.g. 'Infosys Limited')."
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_TICKER_SYSTEM_PROMPT = """\
You are a financial data assistant. Given a research query about a company,
extract the most likely stock ticker symbol and the full company name.

Respond with ONLY valid JSON matching this schema:
{"ticker": "TICKER", "company_name": "Full Company Name"}

Use the primary US exchange ticker when the company is dual-listed.
If you are not confident, use "UNKNOWN" for ticker.
Do not include any text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_ticker(query: str) -> TickerExtraction:
    """Use the LLM to extract a ticker symbol and company name from a free-text query.

    Args:
        query: The original user research query.

    Returns:
        A TickerExtraction with ticker and company_name.

    Raises:
        RuntimeError: Propagates from call_structured if both LLM models fail.
    """
    messages = [
        {"role": "system", "content": _TICKER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Research query: {query}"},
    ]
    return call_with_backoff(
        call_structured,
        messages=messages,
        response_schema=TickerExtraction,
    )


def _fetch_yfinance(ticker: str, company_name: str) -> FinancialData:
    """Fetch financial metrics for a ticker using yfinance.

    Returns a FinancialData with data_available=False (all numerics None) if
    yfinance returns an empty or incomplete info dict.

    Args:
        ticker: Stock ticker symbol.
        company_name: Company name to use as fallback label.

    Returns:
        A FinancialData instance.
    """
    try:
        info: dict = yf.Ticker(ticker).info
    except Exception as exc:
        logger.warning("yfinance raised an exception for ticker %s: %s", ticker, exc)
        info = {}

    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        logger.warning(
            "yfinance returned no usable data for ticker %s. Setting data_available=False.",
            ticker,
        )
        return FinancialData(
            ticker=ticker,
            company_name=company_name,
            data_available=False,
            data_as_of=datetime.now(UTC),
        )

    def _safe_float(key: str) -> float | None:
        """Return info[key] as float, or None if missing/non-numeric."""
        val = info.get(key)
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    revenue_growth = _safe_float("revenueGrowth")
    price = _safe_float("currentPrice") or _safe_float("regularMarketPrice")

    return FinancialData(
        ticker=ticker,
        company_name=info.get("longName", company_name),
        current_price=price,
        pe_ratio=_safe_float("trailingPE"),
        pb_ratio=_safe_float("priceToBook"),
        roe=_safe_float("returnOnEquity"),
        revenue_growth_yoy=revenue_growth,
        debt_to_equity=_safe_float("debtToEquity"),
        market_cap=_safe_float("marketCap"),
        data_as_of=datetime.now(UTC),
        data_available=True,
    )


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


def financial_data_node(state: ResearchState) -> ResearchState:
    """Extract ticker from query, fetch financial metrics, and populate state."""
    query = state.get("query", "")
    job_id = state.get("job_id", "unknown")

    span = create_span(
        "financial_data_node",
        trace_id=state.get("langfuse_trace_id"),
        input_data={"query": query, "job_id": job_id},
    )

    try:
        extraction = _extract_ticker(query)
        ticker = extraction.ticker
        company_name = extraction.company_name

        logger.info("Extracted ticker: %s (%s) for job %s", ticker, company_name, job_id)

        if ticker == "UNKNOWN":
            logger.warning(
                "LLM could not determine ticker for job %s. "
                "financial_data will have data_available=False.",
                job_id,
            )
            financial_data = FinancialData(
                ticker="UNKNOWN",
                company_name=company_name or "Unknown Company",
                data_available=False,
                data_as_of=datetime.now(UTC),
            )
        else:
            logger.info("Fetching yfinance data for ticker=%s, job=%s", ticker, job_id)
            financial_data = _fetch_yfinance(ticker, company_name)

        span.update(output={"ticker": financial_data.ticker, "data_available": financial_data.data_available})
        span.end()

    except Exception as exc:
        error_msg = f"financial_data_node failed: {exc}"
        logger.error(error_msg)
        span.update(level="ERROR", status_message=error_msg)
        span.end()
        return {"error": error_msg}  # type: ignore[return-value]

    return {"financial_data": financial_data}  # type: ignore[return-value]
