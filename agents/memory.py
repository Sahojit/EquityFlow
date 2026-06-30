"""Memory agent — retrieves relevant past research from ChromaDB using local embeddings.

No LLM call is made here. Embeddings are generated locally with sentence-transformers
"all-MiniLM-L6-v2". No LangFuse span is created per spec (data retrieval, not reasoning).
"""

from __future__ import annotations

import logging
import os

import chromadb
from sentence_transformers import SentenceTransformer

from graph.state import ResearchState

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "alpha_agents_memory"
_TOP_K = 3

# Lazy singletons — initialised on first call to avoid slow import at module load
_chroma_client: chromadb.ClientAPI | None = None
_embedder: SentenceTransformer | None = None


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------


def _get_chroma_client() -> chromadb.ClientAPI:
    """Return the ChromaDB persistent client, initialising it on first call.

    Returns:
        A chromadb.PersistentClient pointed at CHROMADB_PATH from the environment.
    """
    global _chroma_client
    if _chroma_client is None:
        path = os.getenv("CHROMADB_PATH", "./chroma_db")
        _chroma_client = chromadb.PersistentClient(path=path)
        logger.info("ChromaDB client initialised at path: %s", path)
    return _chroma_client


def _get_embedder() -> SentenceTransformer:
    """Return the sentence-transformers model, initialising it on first call.

    Returns:
        A SentenceTransformer loaded with "all-MiniLM-L6-v2".
    """
    global _embedder
    if _embedder is None:
        logger.info("Loading sentence-transformers model 'all-MiniLM-L6-v2'…")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedder ready.")
    return _embedder


# ---------------------------------------------------------------------------
# Public store helper (called by the API layer after a note is approved)
# ---------------------------------------------------------------------------


def store_note_in_memory(note_text: str, metadata: dict) -> None:
    """Embed and persist an approved research note into ChromaDB.

    This is called by the API layer after the human approves a note, so future
    queries on the same company benefit from prior research.

    Args:
        note_text: Full markdown text of the research note.
        metadata: Dict of metadata to store alongside the embedding
            (e.g. {"ticker": "INFY", "job_id": "..."}).

    Raises:
        Exception: Any ChromaDB or embedding error is logged and re-raised.
    """
    try:
        client = _get_chroma_client()
        embedder = _get_embedder()
        collection = client.get_or_create_collection(_COLLECTION_NAME)

        embedding: list[float] = embedder.encode(note_text).tolist()
        doc_id = metadata.get("job_id", note_text[:64])

        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[note_text],
            metadatas=[metadata],
        )
        logger.info("Stored note in ChromaDB. id=%s", doc_id)
    except Exception as exc:
        logger.error("Failed to store note in ChromaDB: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


def memory_node(state: ResearchState) -> ResearchState:
    """Retrieve top-3 most relevant past research notes from ChromaDB.

    Embeds the current query locally (no API call), queries ChromaDB, and
    concatenates the top-K results into a single context string. Returns an
    empty string if the collection is empty or ChromaDB is unavailable —
    the writer will simply proceed without memory context.

    Args:
        state: Current pipeline state. Reads ``query`` and ``job_id``.

    Returns:
        Updated state with ``memory_context`` populated (may be empty string).
    """
    query = state.get("query", "")
    job_id = state.get("job_id", "unknown")

    try:
        client = _get_chroma_client()
        embedder = _get_embedder()

        collection = client.get_or_create_collection(_COLLECTION_NAME)

        # If collection is empty, skip retrieval
        count = collection.count()
        if count == 0:
            logger.info(
                "ChromaDB collection '%s' is empty. Returning empty memory context. job=%s",
                _COLLECTION_NAME,
                job_id,
            )
            return {"memory_context": ""}  # type: ignore[return-value]

        query_embedding: list[float] = embedder.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(_TOP_K, count),
        )

        documents: list[str] = results.get("documents", [[]])[0]
        memory_context = "\n\n---\n\n".join(documents)

        logger.info(
            "memory_node retrieved %d past notes for job %s.", len(documents), job_id
        )
        return {"memory_context": memory_context}  # type: ignore[return-value]

    except Exception as exc:
        logger.error(
            "memory_node failed for job %s: %s. Returning empty context.", job_id, exc
        )
        # Non-fatal — writer proceeds without memory context
        return {"memory_context": ""}  # type: ignore[return-value]
