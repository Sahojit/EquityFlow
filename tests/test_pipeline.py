"""Integration test for graph/pipeline.py.

Runs a full LangGraph pipeline with ALL external calls mocked:
- Tavily (web + news)
- yfinance
- OpenAI/HuggingFace client
- ChromaDB
- LangFuse

Assertions:
- Final state contains a valid ResearchNote
- revision_count <= MAX_REVISION_LOOPS
- error is None
- At least one citation URL appears in the note's full_text
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from graph.state import ResearchState

QUERY = "Infosys Q1 2026 outlook"

# ---------------------------------------------------------------------------
# Shared mock payloads
# ---------------------------------------------------------------------------

_ORCHESTRATOR_JSON = json.dumps(
    {
        "sub_questions": [
            "What is Infosys Q1 2026 revenue outlook?",
            "What are key risks for Infosys in 2026?",
            "How does Infosys valuation compare to TCS?",
        ],
        "research_plan": "Analyse Infosys financials, news, and web data.",
    }
)

_SUMMARY_JSON = json.dumps({"summary": "Infosys posted strong Q1 2026 results."})

_TICKER_JSON = json.dumps({"ticker": "INFY", "company_name": "Infosys Limited"})

_NEWS_JSON = json.dumps({"sentiment": "positive", "summary": "Infosys beat Q1 estimates."})

_NOTE_JSON = json.dumps(
    {
        "company_name": "Infosys Limited",
        "ticker": "INFY",
        "analyst": "AlphaAgents v1",
        "date_generated": datetime.now(UTC).isoformat(),
        "investment_thesis": "Infosys is well-positioned for 2026 growth.",
        "key_risks": ["Macro slowdown", "Currency headwinds"],
        "valuation_summary": "Trading at 24x trailing P/E, fair value.",
        "comparable_companies": ["TCS", "WIPRO"],
        "recommendation": "Buy",
        "confidence": "Medium",
        "citations": ["https://example.com/infosys-q1", "https://example.com/infosys-news"],
        "full_text": (
            "# Infosys Limited (INFY)\n\n"
            "## Executive Summary\n"
            "Infosys [beat Q1 estimates](https://example.com/infosys-news).\n\n"
            "## Investment Thesis\nInfosys is well-positioned for 2026 growth "
            "[source](https://example.com/infosys-q1).\n"
        ),
    }
)

_CRITIQUE_JSON = json.dumps(
    {
        "critique": [
            {
                "claim": "Infosys beat Q1 estimates",
                "verdict": "supported",
                "reason": "Backed by citation.",
            }
        ]
    }
)

_YFINANCE_INFO = {
    "longName": "Infosys Limited",
    "currentPrice": 18.5,
    "trailingPE": 24.3,
    "priceToBook": 6.1,
    "returnOnEquity": 0.31,
    "revenueGrowth": 0.07,
    "debtToEquity": 8.5,
    "marketCap": 76_000_000_000,
}

_TAVILY_RESULTS = {
    "results": [
        {
            "url": "https://example.com/infosys-q1",
            "title": "Infosys Q1 2026",
            "content": "Infosys posted strong Q1 2026 results.",
            "source": "Reuters",
            "published_date": "2026-07-14T10:00:00Z",
        }
    ]
}

_NEWS_TAVILY = {
    "results": [
        {
            "title": "Infosys beats Q1 estimates",
            "url": "https://example.com/infosys-news",
            "content": "Infosys beat Q1 2026 expectations.",
            "source": "Bloomberg",
            "published_date": "2026-07-15T08:00:00Z",
        }
    ]
}


def _make_completion(content: str) -> MagicMock:
    """Build a mock ChatCompletion with the given content string."""
    choice = MagicMock()
    choice.message.content = content
    c = MagicMock()
    c.choices = [choice]
    return c


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
    {"TAVILY_API_KEY": "tvly_test", "GROQ_API_KEY": "gsk_test", "CHROMADB_PATH": "/tmp/test_chroma"},
)
async def test_full_pipeline_integration(
    mock_create_span: MagicMock,
    mock_get_client: MagicMock,
    mock_web_tavily_cls: MagicMock,
    mock_news_tavily_cls: MagicMock,
    mock_ticker_cls: MagicMock,
    mock_get_embedder: MagicMock,
    mock_get_chroma: MagicMock,
) -> None:
    """Full pipeline integration test with all external APIs mocked."""
    # Parallel nodes run concurrently, so the LLM call order is non-deterministic.
    # Dispatch by inspecting the system prompt content to identify the caller.
    def _dispatch_llm(messages: list) -> MagicMock:
        """Return the correct mock completion based on the agent's system prompt."""
        system_content = next(
            (m["content"] for m in messages if m.get("role") == "system"), ""
        )
        # Most specific matches first to avoid collisions
        if "compliance reviewer" in system_content or "unsupported" in system_content:
            return _make_completion(_CRITIQUE_JSON)
        if "senior equity research analyst" in system_content:
            return _make_completion(_NOTE_JSON)
        if "orchestrator" in system_content or "research_plan" in system_content:
            return _make_completion(_ORCHESTRATOR_JSON)
        if "news analyst" in system_content or "sentiment" in system_content.lower()[:200]:
            return _make_completion(_NEWS_JSON)
        if "financial data assistant" in system_content:
            return _make_completion(_TICKER_JSON)
        # Default: web summary
        return _make_completion(_SUMMARY_JSON)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = (
        lambda *a, **kw: _dispatch_llm(kw.get("messages", a[0] if a else []))
    )
    mock_get_client.return_value = mock_client

    # Tavily — web researcher
    mock_web_tavily = MagicMock()
    mock_web_tavily.search.return_value = _TAVILY_RESULTS
    mock_web_tavily_cls.return_value = mock_web_tavily

    # Tavily — news agent
    mock_news_tavily = MagicMock()
    mock_news_tavily.search.return_value = _NEWS_TAVILY
    mock_news_tavily_cls.return_value = mock_news_tavily

    # yfinance
    mock_ticker = MagicMock()
    mock_ticker.info = _YFINANCE_INFO
    mock_ticker_cls.return_value = mock_ticker

    # ChromaDB (empty collection — memory returns "")
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    mock_get_chroma.return_value = mock_chroma_client

    # Embedder
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [0.1] * 384
    mock_get_embedder.return_value = mock_embedder

    # create_span returns a no-op span mock
    mock_create_span.return_value = MagicMock()

    # Import here to avoid circular import during patching
    from graph.pipeline import build_graph
    test_pipeline = build_graph()

    initial_state: ResearchState = {
        "query": QUERY,
        "job_id": "integration-test-001",
        "revision_count": 0,
        "hitl_decision": None,
        "final_note": None,
        "error": None,
        "langfuse_trace_id": None,
    }

    final_state: ResearchState = await test_pipeline.ainvoke(initial_state)

    # Core assertions
    assert final_state.get("error") is None, f"Pipeline error: {final_state.get('error')}"

    note = final_state.get("final_note") or final_state.get("draft_note")
    assert note is not None, "No research note produced."
    assert note.ticker == "INFY"
    assert note.recommendation in ("Buy", "Hold", "Sell", "Not Rated")

    revision_count = final_state.get("revision_count", 0)
    assert revision_count <= int(__import__("os").getenv("MAX_REVISION_LOOPS", "2")), (
        f"revision_count={revision_count} exceeds MAX_REVISION_LOOPS"
    )

    # At least one citation URL appears in full_text
    for url in note.citations:
        assert url in note.full_text, f"Citation URL '{url}' missing from full_text."
