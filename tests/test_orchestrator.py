"""Tests for agents/orchestrator.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agents.orchestrator import orchestrator_node
from graph.state import ResearchState


def _base_state() -> ResearchState:
    """Return a minimal ResearchState for orchestrator tests."""
    return ResearchState(
        query="Infosys Q1 FY27 outlook",
        job_id="test-job-001",
        revision_count=0,
        hitl_decision=None,
        final_note=None,
        error=None,
        langfuse_trace_id=None,
    )


@patch("agents.orchestrator.create_span")
@patch("llm.client.get_llm_client")
def test_orchestrator_happy_path(
    mock_get_client: MagicMock, mock_create_span: MagicMock
) -> None:
    """orchestrator_node populates sub_questions and research_plan on valid LLM output."""
    valid_response = json.dumps(
        {
            "sub_questions": [
                "What is Infosys revenue outlook for Q1 FY27?",
                "What are the key risks for Infosys in FY27?",
                "How does Infosys valuation compare to peers?",
            ],
            "research_plan": "Analyse Infosys financials and recent news.",
        }
    )
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = valid_response
    completion = MagicMock()
    completion.choices = [choice]
    mock_client.chat.completions.create.return_value = completion
    mock_get_client.return_value = mock_client
    mock_create_span.return_value = MagicMock()

    result = orchestrator_node(_base_state())

    assert "sub_questions" in result
    assert len(result["sub_questions"]) == 3
    assert "research_plan" in result
    assert result.get("error") is None


@patch("agents.orchestrator.create_span")
@patch("llm.client.get_llm_client")
def test_orchestrator_sets_error_on_llm_failure(
    mock_get_client: MagicMock, mock_create_span: MagicMock
) -> None:
    """orchestrator_node sets state['error'] and does not crash when LLM fails."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("LLM unavailable")
    mock_get_client.return_value = mock_client
    mock_create_span.return_value = MagicMock()

    result = orchestrator_node(_base_state())

    assert result.get("error") is not None
    assert "orchestrator_node failed" in result["error"]
    assert result.get("sub_questions") is None
