"""Single source of truth for all LLM interactions.

All agents must import from here. No agent may instantiate its own OpenAI client
or call the HuggingFace Inference API directly.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, TypeVar

from dotenv import load_dotenv
from langfuse import Langfuse
from openai import BadRequestError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIMARY_MODEL: str = os.getenv(
    "PRIMARY_MODEL", "llama-3.1-8b-instant"
)
FALLBACK_MODEL: str = os.getenv(
    "FALLBACK_MODEL", "llama-3.3-70b-versatile"
)
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1/"
MAX_RETRIES: int = 3
BACKOFF_BASE_SECONDS: float = 2.0

T = TypeVar("T", bound=BaseModel)

# Module-level singleton — avoids reconnecting to LangFuse on every span
_langfuse_client: Langfuse | None = None


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that models sometimes wrap JSON in.

    Handles ```json\\n{...}\\n``` and ```\\n{...}\\n``` patterns.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        # drop optional language tag (e.g. "json")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    return text


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def get_llm_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at the Groq API.

    Reads GROQ_API_KEY from the environment (set in .env).

    Raises:
        RuntimeError: If GROQ_API_KEY is not set in the environment.
    """
    token = os.getenv("GROQ_API_KEY")
    if not token:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://console.groq.com/keys"
        )
    return OpenAI(api_key=token, base_url=GROQ_BASE_URL)


def get_langfuse_client() -> Langfuse:
    """Return the LangFuse client singleton for pipeline observability.

    Reads HF_TOKEN, LANGFUSE_PUBLIC_KEY, and LANGFUSE_SECRET_KEY from the
    environment (set in .env).
    """
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            httpx_client_kwargs={
                "headers": {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
            }
            if os.getenv("HF_TOKEN")
            else None,
        )
    return _langfuse_client


def trace_span(name: str, input_data: dict, output_data: dict) -> None:
    """Log a single named span to LangFuse with its input and output payloads.

    Failures are logged and swallowed — tracing must never crash the pipeline.

    Args:
        name: Span name (typically the calling node's name).
        input_data: Input payload to record on the span.
        output_data: Output payload to record on the span.
    """
    try:
        lf = get_langfuse_client()
        with lf.start_as_current_span(name=name, input=input_data) as span:
            span.update(output=output_data)
    except Exception as exc:
        logger.warning("trace_span failed for '%s': %s. Tracing skipped.", name, exc)


# ---------------------------------------------------------------------------
# Core structured call
# ---------------------------------------------------------------------------


def call_structured(
    *,
    messages: list[dict[str, str]],
    response_schema: type[T],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> T:
    """Call the HuggingFace Inference API and parse the response into a Pydantic model.

    Tries PRIMARY_MODEL first. If the response is not valid JSON or fails Pydantic
    validation, retries once with FALLBACK_MODEL. Raises RuntimeError if both fail.

    Args:
        messages: OpenAI-style chat messages list.
        response_schema: A Pydantic BaseModel subclass to validate the response against.
        model: Override the model. Defaults to PRIMARY_MODEL.
        temperature: Sampling temperature. Lower = more deterministic.
        max_tokens: Maximum tokens in the completion.

    Returns:
        A validated instance of ``response_schema``.

    Raises:
        RuntimeError: If both PRIMARY_MODEL and FALLBACK_MODEL return invalid output.
    """
    client = get_llm_client()
    models_to_try = [model or PRIMARY_MODEL, FALLBACK_MODEL]

    last_error: Exception | None = None

    for attempt_model in models_to_try:
        try:
            logger.debug("Calling model=%s schema=%s", attempt_model, response_schema.__name__)
            completion = client.chat.completions.create(
                model=attempt_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                # No response_format — Groq's server-side JSON validation rejects
                # markdown-fenced output with 400; we strip fences ourselves instead.
            )
            raw = _strip_fences(completion.choices[0].message.content or "")
            logger.debug("Raw response from %s: %s", attempt_model, raw[:300])

            # strict=False allows literal control chars (e.g. real \n inside strings)
            # that models sometimes embed instead of escaped \n sequences.
            parsed_json = json.loads(raw, strict=False)
            return response_schema.model_validate(parsed_json)

        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "Model %s returned invalid output (%s). Trying fallback.",
                attempt_model, exc,
            )
            last_error = exc

        except BadRequestError as exc:
            # Groq json_validate_failed — model output was malformed; try next model.
            logger.warning(
                "Model %s BadRequestError (%s). Trying fallback.", attempt_model, exc
            )
            last_error = exc

        except Exception as exc:
            logger.error("Unexpected error calling model %s: %s", attempt_model, exc)
            raise

    raise RuntimeError(
        f"Both {PRIMARY_MODEL} and {FALLBACK_MODEL} failed to return valid "
        f"{response_schema.__name__} output. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Backoff wrapper
# ---------------------------------------------------------------------------


def call_with_backoff(
    fn: Any,
    *args: Any,
    max_retries: int = MAX_RETRIES,
    **kwargs: Any,
) -> Any:
    """Wrap any callable with exponential backoff on HTTP 429 (rate limit) errors.

    Intended usage::

        result = call_with_backoff(call_structured, messages=..., response_schema=...)

    Args:
        fn: The callable to wrap (typically ``call_structured``).
        *args: Positional arguments forwarded to ``fn``.
        max_retries: Maximum number of retry attempts after the first failure.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        The return value of ``fn``.

    Raises:
        RateLimitError: If the rate limit is still hit after ``max_retries`` attempts.
        Exception: Any non-rate-limit exception raised by ``fn`` is re-raised immediately.
    """
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Rate limit exceeded after %d retries. Giving up.", max_retries
                )
                raise
            wait = BACKOFF_BASE_SECONDS ** attempt
            logger.warning(
                "Rate limit hit (attempt %d/%d). Waiting %.1fs before retry.",
                attempt,
                max_retries,
                wait,
            )
            time.sleep(wait)
