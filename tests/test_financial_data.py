"""Tests for agents/financial_data.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agents.financial_data import financial_data_node
from graph.state import ResearchState


def _base_state() -> ResearchState:
    return ResearchState(
        query="Infosys Q1 FY27 outlook",
        job_id="test-job-003",
        revision_count=0,
        hitl_decision=None,
        final_note=None,
        error=None,
        langfuse_trace_id=None,
    )


@patch("llm.client.get_llm_client")
@patch("agents.financial_data.yf.Ticker")
def test_financial_data_happy_path(
    mock_ticker_cls: MagicMock,
    mock_get_client: MagicMock,
) -> None:
    """financial_data_node returns populated FinancialData when yfinance has data."""
    # Mock LLM ticker extraction
    ticker_json = json.dumps({"ticker": "INFY", "company_name": "Infosys Limited"})
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = ticker_json
    completion = MagicMock()
    completion.choices = [choice]
    mock_client.chat.completions.create.return_value = completion
    mock_get_client.return_value = mock_client

    # Mock yfinance data
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "longName": "Infosys Limited",
        "currentPrice": 18.5,
        "trailingPE": 24.3,
        "priceToBook": 6.1,
        "returnOnEquity": 0.31,
        "revenueGrowth": 0.07,
        "debtToEquity": 8.5,
        "marketCap": 76_000_000_000,
    }
    mock_ticker_cls.return_value = mock_ticker

    result = financial_data_node(_base_state())

    fd = result.get("financial_data")
    assert fd is not None
    assert fd.ticker == "INFY"
    assert fd.data_available is True
    assert fd.current_price == 18.5
    assert fd.pe_ratio == 24.3
    assert result.get("error") is None


@patch("llm.client.get_llm_client")
@patch("agents.financial_data.yf.Ticker")
def test_financial_data_sets_unavailable_when_yfinance_empty(
    mock_ticker_cls: MagicMock,
    mock_get_client: MagicMock,
) -> None:
    """financial_data_node sets data_available=False when yfinance returns empty info."""
    ticker_json = json.dumps({"ticker": "INFY", "company_name": "Infosys Limited"})
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = ticker_json
    completion = MagicMock()
    completion.choices = [choice]
    mock_client.chat.completions.create.return_value = completion
    mock_get_client.return_value = mock_client

    mock_ticker = MagicMock()
    mock_ticker.info = {}  # empty — simulates yfinance returning nothing
    mock_ticker_cls.return_value = mock_ticker

    result = financial_data_node(_base_state())

    fd = result.get("financial_data")
    assert fd is not None
    assert fd.data_available is False
    assert fd.current_price is None
    assert result.get("error") is None


@patch("llm.client.get_llm_client")
@patch("agents.financial_data.yf.Ticker")
def test_financial_data_unknown_ticker(
    mock_ticker_cls: MagicMock,
    mock_get_client: MagicMock,
) -> None:
    """financial_data_node sets data_available=False when LLM returns UNKNOWN ticker."""
    ticker_json = json.dumps({"ticker": "UNKNOWN", "company_name": "Some Company"})
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = ticker_json
    completion = MagicMock()
    completion.choices = [choice]
    mock_client.chat.completions.create.return_value = completion
    mock_get_client.return_value = mock_client

    result = financial_data_node(_base_state())

    fd = result.get("financial_data")
    assert fd is not None
    assert fd.ticker == "UNKNOWN"
    assert fd.data_available is False
    # yfinance should not have been called
    mock_ticker_cls.assert_not_called()
