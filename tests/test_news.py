"""Tests for agents/news.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agents.news import news_node
from graph.state import ResearchState


def _base_state() -> ResearchState:
    return ResearchState(
        query="Infosys Q1 FY27 outlook",
        job_id="test-job-004",
        revision_count=0,
        hitl_decision=None,
        final_note=None,
        error=None,
        langfuse_trace_id=None,
    )


@patch("agents.news.create_span")
@patch("llm.client.get_llm_client")
@patch("agents.news.TavilyClient")
@patch.dict("os.environ", {"TAVILY_API_KEY": "tvly_test"})
def test_news_happy_path(
    mock_tavily_cls: MagicMock,
    mock_get_client: MagicMock,
    mock_create_span: MagicMock,
) -> None:
    """news_node returns populated NewsResult list on success (batch LLM call)."""
    mock_tavily = MagicMock()
    mock_tavily.search.return_value = {
        "results": [
            {
                "title": "Infosys beats Q1 estimates",
                "url": "https://example.com/infosys-q1",
                "content": "Infosys reported better-than-expected Q1 FY27 results.",
                "source": "Reuters",
                "published_date": "2026-07-14T10:00:00Z",
            }
        ]
    }
    mock_tavily_cls.return_value = mock_tavily

    # Batch response: one entry per article
    batch_json = json.dumps({
        "articles": [{"sentiment": "positive", "summary": "Infosys beat Q1 estimates."}]
    })
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = batch_json
    completion = MagicMock()
    completion.choices = [choice]
    mock_client.chat.completions.create.return_value = completion
    mock_get_client.return_value = mock_client
    mock_create_span.return_value = MagicMock()

    result = news_node(_base_state())

    news = result.get("news_results", [])
    assert len(news) == 1
    assert news[0].sentiment == "positive"
    assert news[0].headline == "Infosys beats Q1 estimates"
    assert result.get("error") is None


@patch("agents.news.create_span")
@patch.dict("os.environ", {}, clear=True)
def test_news_fallback_without_tavily_key(mock_create_span: MagicMock) -> None:
    """news_node returns empty list and does not crash when TAVILY_API_KEY is absent."""
    import os
    os.environ.pop("TAVILY_API_KEY", None)
    mock_create_span.return_value = MagicMock()

    result = news_node(_base_state())

    assert result.get("news_results") == []
    assert result.get("error") is None


@patch("agents.news.create_span")
@patch("llm.client.get_llm_client")
@patch("agents.news.TavilyClient")
@patch.dict("os.environ", {"TAVILY_API_KEY": "tvly_test"})
def test_news_returns_empty_on_batch_llm_failure(
    mock_tavily_cls: MagicMock,
    mock_get_client: MagicMock,
    mock_create_span: MagicMock,
) -> None:
    """news_node returns empty list (not a crash) when the batch LLM call fails."""
    mock_tavily = MagicMock()
    mock_tavily.search.return_value = {
        "results": [
            {"title": "Article 1", "url": "https://example.com/1", "content": "Content 1", "source": "Bloomberg"},
            {"title": "Article 2", "url": "https://example.com/2", "content": "Content 2", "source": "CNBC"},
        ]
    }
    mock_tavily_cls.return_value = mock_tavily

    # Both primary and fallback return invalid JSON — batch call fails gracefully
    mock_client = MagicMock()
    bad_choice = MagicMock()
    bad_choice.message.content = "INVALID JSON {"
    bad_completion = MagicMock()
    bad_completion.choices = [bad_choice]
    mock_client.chat.completions.create.return_value = bad_completion
    mock_get_client.return_value = mock_client
    mock_create_span.return_value = MagicMock()

    result = news_node(_base_state())

    # Batch failure → empty list, not an error state
    assert result.get("news_results") == []
    assert result.get("error") is None
