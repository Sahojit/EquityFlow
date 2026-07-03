# AlphaAgents — Architecture (C4 Level 2)

## System Overview

AlphaAgents is a multi-agent equity research pipeline deployed as two services on Render
(FastAPI backend, Streamlit frontend) with all agent reasoning performed via the
HuggingFace Inference API.

---

## Containers

### 1. Streamlit Frontend (`ui/app.py`)

The user-facing interface running on Render as a separate web service. The analyst
types a company or sector query into the "New Research" tab and submits it. The UI
polls the FastAPI backend every 3 seconds and renders the draft note in the "Review"
tab once the pipeline completes. The analyst can approve, edit, or reject the note via
buttons that call the HITL endpoint. Approved notes appear in the "History" tab.

### 2. FastAPI Backend (`api/main.py`)

The REST API that orchestrates the system. It accepts research queries via
`POST /research`, creates a UUID job, persists initial state to SQLite, and launches the
LangGraph pipeline in a background task. Poll endpoints (`GET /research/{job_id}` and
`GET /research/{job_id}/state`) expose job progress. The HITL endpoint
(`POST /research/{job_id}/hitl`) records the analyst's decision and, on approval,
stores the note in ChromaDB for future memory retrieval.

### 3. LangGraph Pipeline (`graph/pipeline.py`)

A compiled `StateGraph` with 8 nodes. The orchestrator node decomposes the query into
sub-questions and a research plan. Four worker nodes (web researcher, financial data,
news, memory) execute in parallel, feeding into the writer. The writer produces a
structured `ResearchNote`. The critic reviews it for unsupported claims. A conditional
router decides whether to loop back to the writer (if >3 unresolved claims and under the
revision cap) or advance to the HITL boundary node.

### 4. LLM Client (`llm/client.py`)

The single source of truth for all model interactions. Wraps the OpenAI Python SDK
pointed at `https://api-inference.huggingface.co/v1/`. Provides `call_structured()`
(returns validated Pydantic models, falls back to secondary model on bad JSON) and
`call_with_backoff()` (exponential backoff on HTTP 429 rate-limit errors).

### 5. ChromaDB (Local Persistent Vector Store)

Stores embeddings of approved research notes. The memory agent queries it at the start
of each pipeline run using local `sentence-transformers` embeddings to surface relevant
prior research on the same company or sector. Persisted to the Render service disk.

### 6. SQLite (Job Metadata Store)

Stores one row per research job: `job_id`, `status`, timestamps, and the full serialised
`ResearchState` as JSON. Used by the API layer for job polling and HITL decision
persistence. Lives on the Render service disk alongside ChromaDB.

### 7. LangFuse (Observability)

Each LLM-facing node creates a LangFuse span at entry and logs input/output keys and
any errors. A trace-level ID is threaded through `ResearchState.langfuse_trace_id` so
all spans for a single pipeline run are grouped in the LangFuse cloud dashboard.

### 8. External Data Sources

- **HuggingFace Inference API** — LLM inference (free tier)
- **Tavily** — Web and news search (free tier, ~1000 req/month)
- **yfinance** — Yahoo Finance financial metrics (non-commercial, no API key)

---

## Data Flow (Single Pipeline Run)

```
User → Streamlit → POST /research → FastAPI → LangGraph Pipeline
                                                    │
                                    ┌───────────────┴──────────────────┐
                                    │          Orchestrator             │
                                    └───────────────┬──────────────────┘
                          ┌─────────┬───────────────┼──────────────┐
                    Web Researcher  Financial Data   News      Memory (ChromaDB)
                          └─────────┴───────────────┼──────────────┘
                                                    │
                                                 Writer (LLM)
                                                    │
                                                 Critic (LLM)
                                                    │
                                          Revision Router (conditional)
                                                    │
                                               HITL Node
                                                    │
                               FastAPI ← GET /state ← Streamlit polls
                                    │
                    Analyst reviews → POST /hitl → SQLite + ChromaDB
```
