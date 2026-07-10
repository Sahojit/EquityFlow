"""End-to-end integration test for the full AlphaAgents LangGraph pipeline.

All external I/O is mocked:
  - Groq LLM (via llm.client.get_llm_client)
  - Tavily web search and news search
  - yfinance
  - ChromaDB + sentence-transformers
  - LangFuse spans

Assertions cover the contract the API and UI depend on:
  - final state produces a valid ResearchNote
  - critique is a list
  - revision_count <= MAX_REVISION_LOOPS
  - error is None
  - citations list is not empty
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from graph.state import ResearchState

QUERY = "Infosys Q1 2026 outlook"

# ---------------------------------------------------------------------------
# Fixture payloads — realistic mock responses for each agent's LLM call
# ---------------------------------------------------------------------------

_ORCHESTRATOR_JSON = json.dumps(
    {
        "sub_questions": [
            "What is Infosys Q1 2026 revenue forecast?",
            "What are key growth drivers for Infosys in FY2026?",
            "How does Infosys valuation compare to TCS and Wipro?",
        ],
        "research_plan": (
            "1. Gather Infosys Q1 2026 financial results and forward guidance. "
            "2. Analyse analyst sentiment and news flow. "
            "3. Compare valuation multiples against Indian IT peers."
        ),
    }
)

# Web researcher — per-result SnippetSummary ({"summary": "..."})
_WEB_SNIPPET_JSON = json.dumps(
    {"summary": "Infosys Q1 2026 revenue grew 8% YoY, beating consensus estimates."}
)

# Financial data — ticker extraction
_TICKER_JSON = json.dumps({"ticker": "INFY", "company_name": "Infosys Limited"})

# News analyst — batched format (matches NewsAnalysisBatch schema)
_NEWS_BATCH_JSON = json.dumps(
    {
        "articles": [
            {
                "sentiment": "positive",
                "summary": "Infosys beat Q1 2026 EPS estimates by 4%, raising full-year guidance.",
            }
        ]
    }
)

# Writer — full ResearchNote
_NOTE_JSON = json.dumps(
    {
        "company_name": "Infosys Limited",
        "ticker": "INFY",
        "analyst": "AlphaAgents v1",
        "date_generated": datetime.now(UTC).isoformat(),
        "investment_thesis": (
            "Infosys is well-positioned for FY2026 driven by large AI and cloud deals. "
            "Margin recovery and improving deal pipeline support a constructive view."
        ),
        "key_risks": [
            "Macro slowdown reducing client IT budgets",
            "USD/INR currency headwinds compressing margins",
            "Attrition and talent cost inflation",
        ],
        "valuation_summary": (
            "Infosys trades at 24x trailing P/E, a 14% discount to TCS. "
            "DCF implies fair value of ~$21, offering ~13% upside [source](https://example.com/infosys-dcf)."
        ),
        "comparable_companies": ["TCS", "WIPRO", "HCL"],
        "recommendation": "Buy",
        "confidence": "Medium",
        "citations": [
            "https://example.com/infosys-q1",
            "https://example.com/infosys-news",
            "https://example.com/infosys-dcf",
        ],
        "full_text": (
            "# Infosys Limited (INFY) — Equity Research Note\n\n"
            "## Executive Summary\n"
            "Infosys delivered a strong Q1 2026, [beating estimates](https://example.com/infosys-q1) "
            "on revenue and EPS. Deal wins in AI and cloud remain robust.\n\n"
            "## Investment Thesis\n"
            "Infosys is well-positioned for FY2026 growth driven by large AI and cloud deals "
            "[source](https://example.com/infosys-news).\n\n"
            "## Financial Analysis\n"
            "Revenue grew 8% YoY. P/E of 24x represents a discount to TCS.\n\n"
            "## Key Risks\n"
            "- Macro slowdown\n- Currency headwinds\n- Talent cost inflation\n\n"
            "## Valuation\n"
            "DCF analysis implies fair value ~$21 [source](https://example.com/infosys-dcf).\n\n"
            "## Recommendation\n"
            "**BUY** with Medium confidence.\n"
        ),
    }
)

# Critic — one supported claim, no unresolved items → routes directly to HITL
_CRITIQUE_JSON = json.dumps(
    {
        "critique": [
            {
                "claim": "Infosys beat Q1 2026 EPS estimates by 4%",
                "verdict": "supported",
                "reason": "Backed by inline citation https://example.com/infosys-q1",
            },
            {
                "claim": "Infosys trades at 24x trailing P/E",
                "verdict": "supported",
                "reason": "Consistent with financial data provided.",
            },
        ]
    }
)

_YFINANCE_INFO = {
    "longName": "Infosys Limited",
    "currentPrice": 18.5,
    "trailingPE": 24.3,
    "priceToBook": 6.1,
    "returnOnEquity": 0.31,
    "revenueGrowth": 0.08,
    "debtToEquity": 8.5,
    "marketCap": 76_000_000_000,
}

_TAVILY_WEB_RESULTS = {
    "results": [
        {
            "url": "https://example.com/infosys-q1",
            "title": "Infosys Q1 2026 Results",
            "content": "Infosys posted strong Q1 2026 results with 8% revenue growth.",
            "source": "Reuters",
            "published_date": "2026-07-14T10:00:00Z",
        }
    ]
}

_TAVILY_NEWS_RESULTS = {
    "results": [
        {
            "title": "Infosys beats Q1 estimates, raises guidance",
            "url": "https://example.com/infosys-news",
            "content": "Infosys beat Q1 2026 EPS estimates by 4% and raised full-year revenue guidance.",
            "source": "Bloomberg",
            "published_date": "2026-07-15T08:00:00Z",
        }
    ]
}


def _make_completion(content: str) -> MagicMock:
    """Build a minimal mock ChatCompletion with the given JSON content string."""
    choice = MagicMock()
    choice.message.content = content
    mock = MagicMock()
    mock.choices = [choice]
    return mock


def _dispatch_llm(messages: list) -> MagicMock:
    """Return the correct mock completion based on the agent's system prompt.

    Ordered most-specific first to avoid false matches.
    """
    system = next(
        (m["content"] for m in messages if m.get("role") == "system"), ""
    )
    if "compliance reviewer" in system or '"unsupported"' in system:
        return _make_completion(_CRITIQUE_JSON)
    if "senior equity research analyst" in system:
        return _make_completion(_NOTE_JSON)
    if "orchestrator" in system or "research_plan" in system:
        return _make_completion(_ORCHESTRATOR_JSON)
    if "news analyst" in system:
        return _make_completion(_NEWS_BATCH_JSON)
    if "financial data assistant" in system:
        return _make_completion(_TICKER_JSON)
    # Default: web researcher per-result snippet summary
    return _make_completion(_WEB_SNIPPET_JSON)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("agents.memory._get_chroma_client")
@patch("agents.memory._get_embedder")
@patch("agents.financial_data.yf.Ticker")
@patch("agents.news.TavilyClient")
@patch("agents.web_researcher.TavilyClient")
@patch("llm.client.get_llm_client")
@patch("llm.tracing.create_span")
@patch.dict(
    "os.environ",
    {
        "TAVILY_API_KEY": "tvly_test",
        "GROQ_API_KEY": "gsk_test",
        "CHROMADB_PATH": "/tmp/test_chroma_integration",
        "MAX_REVISION_LOOPS": "1",
    },
)
async def test_full_pipeline_infosys(
    mock_create_span: MagicMock,
    mock_get_client: MagicMock,
    mock_web_tavily_cls: MagicMock,
    mock_news_tavily_cls: MagicMock,
    mock_ticker_cls: MagicMock,
    mock_get_embedder: MagicMock,
    mock_get_chroma: MagicMock,
) -> None:
    """Full pipeline run on 'Infosys Q1 2026 outlook' with all external calls mocked."""
    # LLM client
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = (
        lambda *a, **kw: _dispatch_llm(kw.get("messages", a[0] if a else []))
    )
    mock_get_client.return_value = mock_client

    # Tavily — web researcher gets its own instance
    mock_web_tavily = MagicMock()
    mock_web_tavily.search.return_value = _TAVILY_WEB_RESULTS
    mock_web_tavily_cls.return_value = mock_web_tavily

    # Tavily — news agent gets its own instance
    mock_news_tavily = MagicMock()
    mock_news_tavily.search.return_value = _TAVILY_NEWS_RESULTS
    mock_news_tavily_cls.return_value = mock_news_tavily

    # yfinance
    mock_ticker = MagicMock()
    mock_ticker.info = _YFINANCE_INFO
    mock_ticker_cls.return_value = mock_ticker

    # ChromaDB — empty collection so memory returns ""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    mock_get_chroma.return_value = mock_chroma_client

    # Sentence-transformers embedder
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [0.0] * 384
    mock_get_embedder.return_value = mock_embedder

    # LangFuse span — no-op
    mock_create_span.return_value = MagicMock()

    from graph.pipeline import build_graph
    graph = build_graph()

    initial_state: ResearchState = {
        "query": QUERY,
        "job_id": "integration-infosys-001",
        "revision_count": 0,
        "hitl_decision": None,
        "final_note": None,
        "error": None,
        "langfuse_trace_id": None,
    }

    final_state: ResearchState = await graph.ainvoke(initial_state)

    # --- error is None ---
    assert final_state.get("error") is None, (
        f"Pipeline raised an error: {final_state.get('error')}"
    )

    # --- final state has a valid ResearchNote ---
    note = final_state.get("final_note") or final_state.get("draft_note")
    assert note is not None, "No ResearchNote produced by the pipeline."
    assert note.ticker == "INFY"
    assert note.company_name == "Infosys Limited"
    assert note.analyst == "AlphaAgents v1"
    assert note.recommendation in ("Buy", "Hold", "Sell", "Not Rated")
    assert note.confidence in ("High", "Medium", "Low")

    # --- critique is a list ---
    critique = final_state.get("critique", [])
    assert isinstance(critique, list), f"Expected critique to be a list, got {type(critique)}"

    # --- revision_count <= MAX_REVISION_LOOPS ---
    import os
    max_loops = int(os.getenv("MAX_REVISION_LOOPS", "1"))
    revision_count = final_state.get("revision_count", 0)
    assert revision_count <= max_loops, (
        f"revision_count={revision_count} exceeds MAX_REVISION_LOOPS={max_loops}"
    )

    # --- citations list is not empty ---
    assert len(note.citations) > 0, "ResearchNote.citations must not be empty."

    # --- at least one citation URL appears in full_text ---
    assert any(url in note.full_text for url in note.citations), (
        "No citation URL found in full_text. All claims must be inline-linked."
    )

    # --- financial data was populated ---
    fin = final_state.get("financial_data")
    assert fin is not None, "financial_data missing from final state."
    assert fin.ticker == "INFY"
    assert fin.data_available is True
    assert fin.current_price == pytest.approx(18.5)
