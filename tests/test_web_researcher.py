"""Tests for agents/web_researcher.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agents.web_researcher import web_researcher_node
from graph.state import ResearchState


def _base_state() -> ResearchState:
    return ResearchState(
        query="Infosys Q1 FY27 outlook",
        job_id="test-job-002",
        sub_questions=[
            "What is Infosys Q1 FY27 revenue guidance?",
            "What are Infosys deal wins in Q1 FY27?",
            "What are the risks to Infosys growth?",
        ],
        revision_count=0,
        hitl_decision=None,
        final_note=None,
        error=None,
        langfuse_trace_id=None,
    )


@patch("agents.web_researcher.trace_span")
@patch("llm.client.get_llm_client")
@patch("agents.web_researcher.TavilyClient")
@patch.dict("os.environ", {"TAVILY_API_KEY": "tvly_test"})
def test_web_researcher_happy_path(
    mock_tavily_cls: MagicMock,
    mock_get_client: MagicMock,
    mock_trace_span: MagicMock,
) -> None:
    """web_researcher_node returns WebResult list on successful search + LLM summary."""
    mock_tavily = MagicMock()
    mock_tavily.search.return_value = {
        "results": [
            {
                "url": "https://example.com/infosys-q1",
                "title": "Infosys Q1 FY27 Results",
                "content": "Infosys reported strong Q1 FY27 revenue growth.",
            }
        ]
    }
    mock_tavily_cls.return_value = mock_tavily

    summary_json = json.dumps({"summary": "Infosys reported strong Q1 revenue."})
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = summary_json
    completion = MagicMock()
    completion.choices = [choice]
    mock_client.chat.completions.create.return_value = completion
    mock_get_client.return_value = mock_client

    result = web_researcher_node(_base_state())

    assert "web_results" in result
    assert len(result["web_results"]) > 0
    assert result["web_results"][0].url == "https://example.com/infosys-q1"
    assert result["web_results"][0].summary == "Infosys reported strong Q1 revenue."
    assert result.get("error") is None


@patch("agents.web_researcher.trace_span")
@patch.dict("os.environ", {}, clear=True)
def test_web_researcher_sets_error_without_tavily_key(
    mock_trace_span: MagicMock,
) -> None:
    """web_researcher_node sets state['error'] and does not crash when TAVILY_API_KEY is absent."""
    result = web_researcher_node(_base_state())

    assert result.get("web_results") is None
    assert result.get("error") is not None
    assert "web_researcher_node failed" in result["error"]


@patch("agents.web_researcher.trace_span")
@patch("llm.client.get_llm_client")
@patch("agents.web_researcher.TavilyClient")
@patch.dict("os.environ", {"TAVILY_API_KEY": "tvly_test"})
def test_web_researcher_skips_failed_questions(
    mock_tavily_cls: MagicMock,
    mock_get_client: MagicMock,
    mock_trace_span: MagicMock,
) -> None:
    """web_researcher_node skips sub-questions where Tavily throws, returns partial results."""
    mock_tavily = MagicMock()
    mock_tavily.search.side_effect = [
        Exception("Tavily timeout"),
        {
            "results": [
                {
                    "url": "https://example.com/infosys-deals",
                    "title": "Infosys Deal Wins",
                    "content": "Infosys won large deals.",
                }
            ]
        },
        {"results": []},
    ]
    mock_tavily_cls.return_value = mock_tavily

    summary_json = json.dumps({"summary": "Infosys won large deals in Q1 FY27."})
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = summary_json
    completion = MagicMock()
    completion.choices = [choice]
    mock_client.chat.completions.create.return_value = completion
    mock_get_client.return_value = mock_client

    result = web_researcher_node(_base_state())

    assert len(result.get("web_results", [])) == 1
    assert result.get("error") is None
