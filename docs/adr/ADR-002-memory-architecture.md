# ADR-002: Memory Architecture — ChromaDB + Local Embeddings

**Status:** Accepted  
**Date:** 2026-06-22

## Context

The pipeline needs to retrieve relevant prior research notes when a user queries
a company that has been analysed before. Requirements:
- Zero incremental API cost for retrieval
- Local persistence across restarts
- Embedding quality sufficient to distinguish between companies/sectors
- No additional infrastructure (no managed vector DB service)

## Decision

**ChromaDB** with **sentence-transformers "all-MiniLM-L6-v2"** (local, CPU-friendly) for
two-tier storage: in-memory during a pipeline run, persisted to disk after HITL approval.

## Consequences

**Positive:**
- Zero infra cost — runs on the same Render instance as the API
- `all-MiniLM-L6-v2` is 22MB, loads in ~2s, produces 384-dim embeddings adequate for company-level similarity
- ChromaDB PersistentClient survives service restarts
- No API key required for embeddings — works fully offline

**Negative:**
- Not horizontally scalable — multiple Render instances would have separate ChromaDB stores
- ChromaDB is not optimised for >1M documents (irrelevant at demo scale)
- Embedding quality is lower than text-embedding-3-large (OpenAI) — acceptable for prototype

## Alternatives Considered

| Option | Reason rejected |
|--------|----------------|
| **Pinecone** | Managed, reliable — but costs money; overkill for demo scale |
| **pgvector** | Requires PostgreSQL with pgvector extension; complex setup on Render free tier |
| **FAISS** | In-memory only — no persistence without custom serialisation |
| **OpenAI text-embedding** | Costs money per embedding; adds API dependency |
