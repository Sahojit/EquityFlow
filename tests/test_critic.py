"""Tests for agents/critic.py."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from agents.critic import critic_node
from graph.state import ResearchNote, ResearchState


def _make_draft_note() -> ResearchNote:
    return ResearchNote(
        company_name="Infosys Limited",
        ticker="INFY",
        analyst="AlphaAgents v1",
        date_generated=datetime.now(UTC),
        investment_thesis="Infosys is well-positioned for FY27 growth.",
        key_risks=["Macro slowdown"],
        valuation_summary="Trading at 24x P/E.",
        comparable_companies=["TCS"],
        recommendation="Buy",
        confidence="Medium",
        citations=["https://example.com/infosys"],
        full_text="# Infosys\n\nInfosys revenue grew 7% YoY [source](https://example.com/infosys).",
    )


def _base_state() -> ResearchState:
    return ResearchState(
        query="Infosys Q1 FY27 outlook",
        job_id="test-job-006",
        draft_note=_make_draft_note(),
        revision_count=1,
        critique=[],
        hitl_decision=None,
        final_note=None,
        error=None,
        langfuse_trace_id=None,
    )


@patch("agents.critic.create_span")
@patch("llm.client.get_llm_client")
def test_critic_happy_path(
    mock_get_client: MagicMock, mock_create_span: MagicMock
) -> None:
    """critic_node returns a populated critique list on successful LLM call."""
    critique_json = json.dumps(
        {
            "critique": [
                {
                    "claim": "Infosys revenue grew 7% YoY",
                    "verdict": "supported",
                    "reason": "Backed by inline citation.",
                },
                {
                    "claim": "Infosys will win 5 mega deals",
                    "verdict": "unsupported",
                    "reason": "No source cited for this projection.",
                },
            ]
        }
    )
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = critique_json
    completion = MagicMock()
    completion.choices = [choice]
    mock_client.chat.completions.create.return_value = completion
    mock_get_client.return_value = mock_client
    mock_create_span.return_value = MagicMock()

    result = critic_node(_base_state())

    critique = result.get("critique", [])
    assert len(critique) == 2
    assert critique[0].verdict == "supported"
    assert critique[1].verdict == "unsupported"
    assert result.get("error") is None


@patch("agents.critic.create_span")
@patch("llm.client.get_llm_client")
def test_critic_returns_empty_list_on_llm_failure(
    mock_get_client: MagicMock, mock_create_span: MagicMock
) -> None:
    """critic_node returns empty critique (not an error) when LLM fails."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("LLM failure")
    mock_get_client.return_value = mock_client
    mock_create_span.return_value = MagicMock()

    result = critic_node(_base_state())

    assert result.get("critique") == []
    assert result.get("error") is None


@patch("agents.critic.create_span")
def test_critic_handles_missing_draft_note(mock_create_span: MagicMock) -> None:
    """critic_node returns empty critique when draft_note is absent."""
    mock_create_span.return_value = MagicMock()
    state = ResearchState(
        query="test",
        job_id="test-job-007",
        revision_count=0,
        critique=[],
        hitl_decision=None,
        final_note=None,
        error=None,
        langfuse_trace_id=None,
    )

    result = critic_node(state)

    assert result.get("critique") == []


@patch("agents.critic.create_span")
@patch("llm.client.get_llm_client")
def test_critic_all_verdict_types(
    mock_get_client: MagicMock, mock_create_span: MagicMock
) -> None:
    """critic_node correctly parses all three verdict values: supported, unsupported, missing_citation."""
    critique_json = json.dumps(
        {
            "critique": [
                {"claim": "Revenue grew 8% YoY", "verdict": "supported", "reason": "Cited inline."},
                {"claim": "Infosys will win 10 deals", "verdict": "unsupported", "reason": "No source."},
                {"claim": "Market cap is $80B", "verdict": "missing_citation", "reason": "Fact with no URL."},
            ]
        }
    )
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = critique_json
    completion = MagicMock()
    completion.choices = [choice]
    mock_client.chat.completions.create.return_value = completion
    mock_get_client.return_value = mock_client
    mock_create_span.return_value = MagicMock()

    result = critic_node(_base_state())

    critique = result.get("critique", [])
    verdicts = {c.verdict for c in critique}
    assert "supported" in verdicts
    assert "unsupported" in verdicts
    assert "missing_citation" in verdicts
    assert result.get("error") is None


@patch("agents.critic.create_span")
@patch("llm.client.get_llm_client")
def test_critic_returns_empty_list_when_note_is_well_cited(
    mock_get_client: MagicMock, mock_create_span: MagicMock
) -> None:
    """critic_node accepts an empty critique list when all claims are well-supported."""
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = json.dumps({"critique": []})
    completion = MagicMock()
    completion.choices = [choice]
    mock_client.chat.completions.create.return_value = completion
    mock_get_client.return_value = mock_client
    mock_create_span.return_value = MagicMock()

    result = critic_node(_base_state())

    assert result.get("critique") == []
    assert result.get("error") is None
