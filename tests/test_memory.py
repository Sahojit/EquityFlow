"""Tests for agents/memory.py.

All external dependencies (ChromaDB, sentence-transformers, LangFuse) are mocked.
No real filesystem, network, or model I/O is performed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.memory import memory_node, store_note_in_memory
from graph.state import ResearchState


def _base_state() -> ResearchState:
    """Return a minimal ResearchState for memory agent tests."""
    return ResearchState(
        query="Infosys Q1 FY27 outlook",
        job_id="test-job-mem-001",
        revision_count=0,
        hitl_decision=None,
        final_note=None,
        error=None,
        langfuse_trace_id=None,
    )


# ---------------------------------------------------------------------------
# Happy path — collection has past notes
# ---------------------------------------------------------------------------


@patch("agents.memory.create_span")
@patch("agents.memory._get_embedder")
@patch("agents.memory._get_chroma_client")
def test_memory_happy_path(
    mock_get_chroma: MagicMock,
    mock_get_embedder: MagicMock,
    mock_create_span: MagicMock,
) -> None:
    """memory_node returns memory_context string built from past ChromaDB documents."""
    past_note_1 = "# Infosys FY26 Research Note\nInfosys posted record FY26 revenue."
    past_note_2 = "# Infosys H2 FY26\nMargins expanded 200bps in H2 FY26."

    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "documents": [[past_note_1, past_note_2]]
    }
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    mock_get_chroma.return_value = mock_chroma_client

    mock_embedding = MagicMock()
    mock_embedding.tolist.return_value = [0.1] * 384
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = mock_embedding
    mock_get_embedder.return_value = mock_embedder

    mock_create_span.return_value = MagicMock()

    result = memory_node(_base_state())

    memory_context = result.get("memory_context", "")
    assert past_note_1 in memory_context
    assert past_note_2 in memory_context
    assert "---" in memory_context  # separator between notes
    assert result.get("error") is None


# ---------------------------------------------------------------------------
# Empty collection path
# ---------------------------------------------------------------------------


@patch("agents.memory.create_span")
@patch("agents.memory._get_embedder")
@patch("agents.memory._get_chroma_client")
def test_memory_empty_collection(
    mock_get_chroma: MagicMock,
    mock_get_embedder: MagicMock,
    mock_create_span: MagicMock,
) -> None:
    """memory_node returns empty string when ChromaDB collection has no documents."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    mock_get_chroma.return_value = mock_chroma_client

    mock_get_embedder.return_value = MagicMock()
    mock_create_span.return_value = MagicMock()

    result = memory_node(_base_state())

    assert result.get("memory_context") == ""
    assert result.get("error") is None
    # Should not have called .query() — collection was empty
    mock_collection.query.assert_not_called()


# ---------------------------------------------------------------------------
# ChromaDB failure — non-fatal fallback
# ---------------------------------------------------------------------------


@patch("agents.memory.create_span")
@patch("agents.memory._get_embedder")
@patch("agents.memory._get_chroma_client")
def test_memory_fallback_on_chroma_error(
    mock_get_chroma: MagicMock,
    mock_get_embedder: MagicMock,
    mock_create_span: MagicMock,
) -> None:
    """memory_node returns empty string (not a crash) when ChromaDB raises."""
    mock_get_chroma.side_effect = RuntimeError("ChromaDB unavailable")
    mock_get_embedder.return_value = MagicMock()
    mock_create_span.return_value = MagicMock()

    result = memory_node(_base_state())

    assert result.get("memory_context") == ""
    assert result.get("error") is None  # memory failure is non-fatal


# ---------------------------------------------------------------------------
# Embedding is called with the query
# ---------------------------------------------------------------------------


@patch("agents.memory.create_span")
@patch("agents.memory._get_embedder")
@patch("agents.memory._get_chroma_client")
def test_memory_embeds_query(
    mock_get_chroma: MagicMock,
    mock_get_embedder: MagicMock,
    mock_create_span: MagicMock,
) -> None:
    """memory_node calls encode() with the query string to produce the search embedding."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {"documents": [["Some past note."]]}
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    mock_get_chroma.return_value = mock_chroma_client

    mock_embedding = MagicMock()
    mock_embedding.tolist.return_value = [0.5] * 384
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = mock_embedding
    mock_get_embedder.return_value = mock_embedder

    mock_create_span.return_value = MagicMock()

    state = _base_state()
    memory_node(state)

    mock_embedder.encode.assert_called_once_with(state["query"])


# ---------------------------------------------------------------------------
# store_note_in_memory
# ---------------------------------------------------------------------------


@patch("agents.memory._get_embedder")
@patch("agents.memory._get_chroma_client")
def test_store_note_in_memory_upserts_document(
    mock_get_chroma: MagicMock,
    mock_get_embedder: MagicMock,
) -> None:
    """store_note_in_memory embeds the note text and upserts it into ChromaDB."""
    mock_collection = MagicMock()
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    mock_get_chroma.return_value = mock_chroma_client

    mock_embedding = MagicMock()
    mock_embedding.tolist.return_value = [0.3] * 384
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = mock_embedding
    mock_get_embedder.return_value = mock_embedder

    note_text = "# Infosys Note\nBuy recommendation."
    metadata = {"job_id": "job-abc", "ticker": "INFY", "company_name": "Infosys Limited"}

    store_note_in_memory(note_text, metadata)

    mock_collection.upsert.assert_called_once()
    call_kwargs = mock_collection.upsert.call_args.kwargs
    assert call_kwargs["ids"] == ["job-abc"]
    assert call_kwargs["documents"] == [note_text]
    assert call_kwargs["metadatas"] == [metadata]
