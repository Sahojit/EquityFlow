"""LangFuse tracing compatibility shim.

Abstracts the LangFuse API so agents work with both v2 and v4. All agents
import from here — never from langfuse directly.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Lazy singleton — initialised on first call
_lf: Any = None


def _get_lf() -> Any:
    """Return the LangFuse client singleton, initialising it on first call."""
    global _lf
    if _lf is None:
        try:
            from langfuse import Langfuse
            _lf = Langfuse()
        except Exception as exc:
            logger.warning("LangFuse client init failed: %s. Tracing disabled.", exc)
            _lf = _NoOpClient()
    return _lf


# ---------------------------------------------------------------------------
# No-op fallbacks
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """No-op span used when LangFuse is unavailable or disabled."""

    def update(self, **kwargs: Any) -> None:
        """Accept any update kwargs silently."""

    def end(self) -> None:
        """No-op end."""


class _NoOpClient:
    """No-op LangFuse client used when LangFuse is unavailable."""

    def create_trace_id(self) -> str:
        """Return a random UUID as a trace ID."""
        return str(uuid.uuid4())

    def start_observation(self, **kwargs: Any) -> _NoOpSpan:
        """Return a no-op span."""
        return _NoOpSpan()

    def trace(self, **kwargs: Any) -> Any:
        """Return a no-op trace."""
        return type("NoOpTrace", (), {"id": str(uuid.uuid4())})()


# ---------------------------------------------------------------------------
# Public helpers used by agents and the API
# ---------------------------------------------------------------------------


def create_span(
    name: str,
    trace_id: str | None = None,
    input_data: dict | None = None,
) -> Any:
    """Create and return a LangFuse span (or a no-op span on failure).

    Compatible with LangFuse v2 and v4.

    Args:
        name: Span name (typically the node name).
        trace_id: Optional LangFuse trace ID to attach this span to.
        input_data: Optional dict of input metadata to log.

    Returns:
        A span object with ``.update(**kwargs)`` and ``.end()`` methods.
    """
    lf = _get_lf()
    try:
        # LangFuse v4 API
        if hasattr(lf, "start_observation"):
            kwargs: dict[str, Any] = {"name": name, "as_type": "span"}
            if input_data:
                kwargs["input"] = input_data
            if trace_id:
                try:
                    from langfuse.types import TraceContext
                    kwargs["trace_context"] = TraceContext(trace_id=trace_id)
                except ImportError:
                    pass  # trace_context attachment best-effort only
            return lf.start_observation(**kwargs)

        # LangFuse v2 API
        if hasattr(lf, "span"):
            return lf.span(
                name=name,
                trace_id=trace_id,
                input=input_data or {},
            )
    except Exception as exc:
        logger.warning("create_span failed for '%s': %s. Using no-op span.", name, exc)

    return _NoOpSpan()


def create_trace_id() -> str:
    """Generate a new LangFuse trace ID (or a plain UUID on failure).

    Returns:
        A trace ID string.
    """
    lf = _get_lf()
    try:
        if hasattr(lf, "create_trace_id"):
            return lf.create_trace_id()
        # v2 fallback
        if hasattr(lf, "trace"):
            t = lf.trace(name="alpha_agents_pipeline")
            return t.id
    except Exception as exc:
        logger.warning("create_trace_id failed: %s. Using random UUID.", exc)
    return str(uuid.uuid4())
