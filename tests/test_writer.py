"""Tests for agents/writer.py."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from agents.writer import writer_node
from graph.state import FinancialData, NewsResult, ResearchState, WebResult


def _make_web_result() -> WebResult:
    return WebResult(
        url="https://example.com/infosys",
        title="Infosys Q1 FY27",
        snippet="Infosys reported strong growth.",
        summary="Infosys posted strong Q1 FY27 results.",
        retrieved_at=datetime.now(UTC),
    )


def _make_financial_data(available: bool = True) -> FinancialData:
    if available:
        return FinancialData(
            ticker="INFY",
            company_name="Infosys Limited",
            current_price=18.5,
            pe_ratio=24.3,
            data_available=True,
            data_as_of=datetime.now(UTC),
        )
    return FinancialData(
        ticker="INFY",
        company_name="Infosys Limited",
        data_available=False,
        data_as_of=datetime.now(UTC),
    )


def _make_news_result() -> NewsResult:
    return NewsResult(
        headline="Infosys beats Q1 estimates",
        source_name="Reuters",
        url="https://example.com/infosys-news",
        published_at=datetime.now(UTC),
        sentiment="positive",
        summary="Infosys beat Q1 FY27 estimates.",
    )


def _base_state() -> ResearchState:
    return ResearchState(
        query="Infosys Q1 FY27 outlook",
        job_id="test-job-005",
        sub_questions=["What is Infosys revenue outlook?"],
        research_plan="Analyse Infosys financials and news.",
        web_results=[_make_web_result()],
        financial_data=_make_financial_data(),
        news_results=[_make_news_result()],
        memory_context="",
        critique=[],
        revision_count=0,
        hitl_decision=None,
        final_note=None,
        error=None,
        langfuse_trace_id=None,
    )


def _valid_note_json() -> str:
    return json.dumps(
        {
            "company_name": "Infosys Limited",
            "ticker": "INFY",
            "analyst": "AlphaAgents v1",
            "date_generated": datetime.now(UTC).isoformat(),
            "investment_thesis": "Infosys is well-positioned for FY27 growth.",
            "key_risks": ["Macro slowdown", "Client budget cuts"],
            "valuation_summary": "Trading at 24x trailing P/E, in line with peers.",
            "comparable_companies": ["TCS", "WIPRO", "HCL"],
            "recommendation": "Buy",
            "confidence": "Medium",
            "citations": ["https://example.com/infosys", "https://example.com/infosys-news"],
            "full_text": (
                "# Infosys Limited (INFY)\n\n"
                "## Executive Summary\nInfosys [beat Q1 estimates](https://example.com/infosys-news).\n\n"
                "## Investment Thesis\nInfosys is well-positioned for FY27 growth.\n"
            ),
        }
    )


@patch("agents.writer.create_span")
@patch("llm.client.get_llm_client")
def test_writer_happy_path(
    mock_get_client: MagicMock, mock_create_span: MagicMock
) -> None:
    """writer_node returns a valid ResearchNote on successful LLM call."""
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = _valid_note_json()
    completion = MagicMock()
    completion.choices = [choice]
    mock_client.chat.completions.create.return_value = completion
    mock_get_client.return_value = mock_client
    mock_create_span.return_value = MagicMock()

    result = writer_node(_base_state())

    note = result.get("draft_note")
    assert note is not None
    assert note.ticker == "INFY"
    assert note.recommendation == "Buy"
    assert len(note.citations) == 2
    assert result.get("revision_count") == 1
    assert result.get("error") is None


@patch("agents.writer.create_span")
@patch("llm.client.get_llm_client")
def test_writer_increments_revision_count(
    mock_get_client: MagicMock, mock_create_span: MagicMock
) -> None:
    """writer_node increments revision_count each time it is called."""
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = _valid_note_json()
    completion = MagicMock()
    completion.choices = [choice]
    mock_client.chat.completions.create.return_value = completion
    mock_get_client.return_value = mock_client
    mock_create_span.return_value = MagicMock()

    state = _base_state()
    state["revision_count"] = 1
    result = writer_node(state)

    assert result.get("revision_count") == 2


@patch("agents.writer.create_span")
@patch("llm.client.get_llm_client")
def test_writer_sets_error_on_llm_failure(
    mock_get_client: MagicMock, mock_create_span: MagicMock
) -> None:
    """writer_node sets state['error'] and does not crash when LLM fails."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("LLM failure")
    mock_get_client.return_value = mock_client
    mock_create_span.return_value = MagicMock()

    result = writer_node(_base_state())

    assert result.get("error") is not None
    assert "writer_node failed" in result["error"]
