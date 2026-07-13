"""Tests for llm/client.py.

All tests mock the OpenAI client — no real API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from openai import RateLimitError
from pydantic import BaseModel

from llm.client import (
    FALLBACK_MODEL,
    PRIMARY_MODEL,
    call_structured,
    call_with_backoff,
    get_llm_client,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _SampleSchema(BaseModel):
    """Minimal Pydantic model used across tests."""

    name: str
    value: int


def _make_completion(content: str) -> MagicMock:
    """Build a mock OpenAI ChatCompletion object with given content string."""
    choice = MagicMock()
    choice.message.content = content
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _make_rate_limit_error() -> RateLimitError:
    """Construct a RateLimitError compatible with openai>=1.x."""
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    return RateLimitError("rate limit", response=response, body={})


# ---------------------------------------------------------------------------
# get_llm_client
# ---------------------------------------------------------------------------


def test_get_llm_client_raises_without_token() -> None:
    """get_llm_client() must raise RuntimeError when GROQ_API_KEY is not set."""
    with patch.dict("os.environ", {}, clear=True):
        import os
        os.environ.pop("GROQ_API_KEY", None)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            get_llm_client()


def test_get_llm_client_returns_openai_client() -> None:
    """get_llm_client() must return an OpenAI client when GROQ_API_KEY is present."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_token"}):
        client = get_llm_client()
        assert client is not None
        assert "groq" in client.base_url.host


# ---------------------------------------------------------------------------
# call_structured — happy path
# ---------------------------------------------------------------------------


@patch("llm.client.get_llm_client")
def test_call_structured_happy_path(mock_get_client: MagicMock) -> None:
    """call_structured returns correct Pydantic model on valid JSON from PRIMARY_MODEL."""
    valid_json = json.dumps({"name": "Infosys", "value": 42})
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_completion(valid_json)
    mock_get_client.return_value = mock_client

    result = call_structured(
        messages=[{"role": "user", "content": "test"}],
        response_schema=_SampleSchema,
    )

    assert isinstance(result, _SampleSchema)
    assert result.name == "Infosys"
    assert result.value == 42
    # Should have used the PRIMARY_MODEL on the first attempt
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == PRIMARY_MODEL


# ---------------------------------------------------------------------------
# call_structured — fallback on invalid JSON
# ---------------------------------------------------------------------------


@patch("llm.client.get_llm_client")
def test_call_structured_retries_with_fallback_on_invalid_json(
    mock_get_client: MagicMock,
) -> None:
    """call_structured retries with FALLBACK_MODEL when PRIMARY_MODEL returns invalid JSON."""
    valid_json = json.dumps({"name": "TCS", "value": 7})
    mock_client = MagicMock()
    # First call (PRIMARY_MODEL) → invalid JSON; second call (FALLBACK_MODEL) → valid
    mock_client.chat.completions.create.side_effect = [
        _make_completion("not-valid-json{{"),
        _make_completion(valid_json),
    ]
    mock_get_client.return_value = mock_client

    result = call_structured(
        messages=[{"role": "user", "content": "test"}],
        response_schema=_SampleSchema,
    )

    assert result.name == "TCS"
    assert mock_client.chat.completions.create.call_count == 2
    # Second call must use the FALLBACK_MODEL
    second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert second_call_kwargs["model"] == FALLBACK_MODEL


# ---------------------------------------------------------------------------
# call_structured — both models fail
# ---------------------------------------------------------------------------


@patch("llm.client.get_llm_client")
def test_call_structured_raises_when_both_models_fail(
    mock_get_client: MagicMock,
) -> None:
    """call_structured raises RuntimeError when both PRIMARY and FALLBACK return bad JSON."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_completion("GARBAGE")
    mock_get_client.return_value = mock_client

    with pytest.raises(RuntimeError, match="failed to return valid"):
        call_structured(
            messages=[{"role": "user", "content": "test"}],
            response_schema=_SampleSchema,
        )

    assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# call_with_backoff — retries on 429
# ---------------------------------------------------------------------------


def test_call_with_backoff_retries_on_rate_limit() -> None:
    """call_with_backoff retries the wrapped function on RateLimitError."""
    call_count = 0

    def _flaky_fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _make_rate_limit_error()
        return "success"

    with patch("llm.client.time.sleep"):  # skip real sleep
        result = call_with_backoff(_flaky_fn, max_retries=3)

    assert result == "success"
    assert call_count == 3


def test_call_with_backoff_raises_after_max_retries() -> None:
    """call_with_backoff raises RateLimitError after exhausting max_retries."""

    def _always_rate_limited() -> None:
        raise _make_rate_limit_error()

    with patch("llm.client.time.sleep"), pytest.raises(RateLimitError):
        call_with_backoff(_always_rate_limited, max_retries=2)


def test_call_with_backoff_does_not_catch_other_exceptions() -> None:
    """call_with_backoff re-raises non-RateLimitError exceptions immediately."""

    def _raises_value_error() -> None:
        raise ValueError("unexpected error")

    with pytest.raises(ValueError, match="unexpected error"):
        call_with_backoff(_raises_value_error)


# ---------------------------------------------------------------------------
# _strip_fences — markdown code-fence removal
# ---------------------------------------------------------------------------


def test_strip_fences_removes_json_fence() -> None:
    """_strip_fences strips ```json ... ``` wrappers that models sometimes emit."""
    from llm.client import _strip_fences

    raw = "```json\n{\"key\": \"value\"}\n```"
    assert _strip_fences(raw) == '{"key": "value"}'


def test_strip_fences_removes_plain_fence() -> None:
    """_strip_fences strips plain ``` ... ``` wrappers without a language tag."""
    from llm.client import _strip_fences

    raw = "```\n{\"key\": 1}\n```"
    assert _strip_fences(raw) == '{"key": 1}'


def test_strip_fences_is_noop_on_clean_json() -> None:
    """_strip_fences leaves already-clean JSON unchanged."""
    from llm.client import _strip_fences

    raw = '{"key": "value"}'
    assert _strip_fences(raw) == raw


# ---------------------------------------------------------------------------
# call_structured — markdown-fenced JSON is accepted
# ---------------------------------------------------------------------------


@patch("llm.client.get_llm_client")
def test_call_structured_accepts_fenced_json(mock_get_client: MagicMock) -> None:
    """call_structured succeeds when the model wraps JSON in ```json fences."""
    fenced = "```json\n{\"name\": \"Wipro\", \"value\": 99}\n```"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_completion(fenced)
    mock_get_client.return_value = mock_client

    result = call_structured(
        messages=[{"role": "user", "content": "test"}],
        response_schema=_SampleSchema,
    )

    assert result.name == "Wipro"
    assert result.value == 99
