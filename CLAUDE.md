# CLAUDE.md — AlphaAgents: Agentic Equity Research Analyst

## Project
Multi-agent LLM pipeline (LangGraph) that produces cited, structured equity research notes with a mandatory human-in-the-loop approval gate. Built during an LLM Systems / Applied GenAI internship (E1 — Agentic Research Analyst track).
Python 3.11, LangGraph, FastAPI, Streamlit, ChromaDB, SQLite, uv.

## Working Directory
`/Users/sahojitkarmakar/Documents/Internship II/Alphaagents/alpha-agents`

## Commit Status (as of 2026-06-30)
Only **2 commits** so far: initial scaffold (`graph/`, `llm/client.py`, README, CI, pyproject) and agent implementations (`agents/*.py`). A substantial amount of work is present but **uncommitted**: `api/`, `docs/`, `eval/`, `ui/`, `llm/tracing.py`, `render.yaml`, `tests/`, `uv.lock`. Do not assume these are stable/final until committed.

## Architecture
```
Query → Orchestrator → [Web Researcher | Financial Data | News | Memory] (parallel fan-out)
      → Writer → Critic → Revision Router
                              ├─ writer_node (if >6 unsupported/missing-citation claims AND revision_count < MAX_REVISION_LOOPS)
                              └─ hitl_node → END
```
Graph defined in `graph/pipeline.py` (`build_graph()`), compiled module-level as `pipeline`. State shape is `graph/state.py::ResearchState` (a `TypedDict(total=False)` — every node writes only the keys it owns and returns `{**state, ...}`).

## Key Rules

### Never do this
- Do not trust the README's tech-stack table for the LLM provider — it says "HuggingFace Inference API" but `llm/client.py` actually calls **Groq** (`https://api.groq.com/openai/v1/`) via an OpenAI-compatible client. Primary model `llama-3.1-8b-instant`, fallback `llama-3.3-70b-versatile`.
- Do not instantiate an OpenAI/Groq client or call the LLM API directly from an agent — `llm/client.py` is the single source of truth; all agents must go through `call_structured()` / `call_with_backoff()`.
- Do not import agent node functions at the top of `graph/pipeline.py` — they're imported lazily inside `build_graph()` to avoid circular imports with `graph.state`.
- Do not pass `response_format` to the Groq chat completion call — Groq's server-side JSON validation rejects markdown-fenced output with a 400; fences are stripped manually via `_strip_fences()` instead.
- Do not use `json.loads(raw)` without `strict=False` when parsing LLM output — models sometimes embed literal control characters instead of escaped `\n`.
- Do not let an agent node raise uncaught — every node must catch exceptions, set `state["error"]`, and return without crashing the pipeline (see `orchestrator_node`).

### Always do this
- Add new Pydantic sub-models for pipeline data to `graph/state.py` only — nothing outside that module defines new state shapes.
- Wrap every LLM call as `call_with_backoff(call_structured, messages=..., response_schema=...)` for exponential backoff on Groq 429s.
- Start a LangFuse span (`create_span` from `llm/tracing.py`) at the top of every node and `.end()` it in both success and error paths.
- Set `data_available=False` on `FinancialData` when yfinance returns no data, rather than leaving numeric fields as misleading zeros.
- Revision loop cap is `MAX_REVISION_LOOPS` (env, default `1`) — respect it in any change to `revision_router` to avoid infinite writer↔critic loops.

## Revision Router Logic
```
unresolved = critique items with verdict in {unsupported, missing_citation}
IF len(unresolved) > 6 AND revision_count < MAX_REVISION_LOOPS → back to writer_node
ELSE                                                            → hitl_node → END
```

## ResearchNote Schema (`graph/state.py`)
```python
company_name, ticker, analyst ("AlphaAgents v1"), date_generated
investment_thesis: str
key_risks: list[str]
valuation_summary: str
comparable_companies: list[str]
recommendation: "Buy" | "Hold" | "Sell" | "Not Rated"
confidence: "High" | "Medium" | "Low"
citations: list[str]
full_text: str
```

## Environment Variables (`.env.example`)
| Var | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | — | required, no default fallback |
| `TAVILY_API_KEY` | — | web/news search |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | — / — / `https://cloud.langfuse.com` | observability |
| `DATABASE_URL` | `sqlite:///./alpha_agents.db` | job tracking + HITL persistence |
| `CHROMADB_PATH` | `./chroma_db` | vector memory |
| `PRIMARY_MODEL` | `llama-3.1-8b-instant` | |
| `FALLBACK_MODEL` | `llama-3.3-70b-versatile` | |
| `MAX_REVISION_LOOPS` | `1` | writer↔critic loop cap |

## Local Dev Setup
```bash
uv sync --all-extras
cp .env.example .env   # fill in GROQ_API_KEY, TAVILY_API_KEY, LANGFUSE keys
make run-api           # FastAPI backend
make run-ui             # Streamlit frontend (localhost:8501)
make test               # pytest; make lint (ruff); make typecheck (mypy strict)
```

## Deployment
`render.yaml` (uncommitted as of last check) targets Render free tier: FastAPI + Streamlit as two separate web services. Cold starts after 15 min inactivity (~30s first request).

## Known Limitations (from README)
- yfinance has data gaps for many international tickers (`data_available=False` flags this)
- Llama-3.1-8B can hallucinate plausible-but-false financial figures — critic + HITL are mitigations, not guarantees
- Groq free tier ~10 req/min — parallel agent fan-out can hit this; `call_with_backoff()` retries but adds latency
- yfinance data is non-commercial-use only — research/educational purposes only

## Module Quick Reference
| File | Purpose |
|---|---|
| `graph/state.py` | `ResearchState` TypedDict + all Pydantic sub-models (WebResult, FinancialData, NewsResult, CritiqueItem, ResearchNote) |
| `graph/pipeline.py` | StateGraph construction, `revision_router`, `hitl_node`, compiled `pipeline` |
| `llm/client.py` | Groq client factory, `call_structured()` (structured JSON output + fallback model), `call_with_backoff()` |
| `llm/tracing.py` | LangFuse span helpers (uncommitted) |
| `agents/orchestrator.py` | Decomposes query into sub-questions + research plan |
| `agents/web_researcher.py` | Tavily web search + LLM summarization |
| `agents/financial_data.py` | yfinance metrics fetch |
| `agents/news.py` | News search + sentiment/summary extraction |
| `agents/memory.py` | ChromaDB retrieval of past research |
| `agents/writer.py` | Synthesizes `ResearchNote` from all agent outputs |
| `agents/critic.py` | LLM-as-judge claim verification, produces `CritiqueItem` list |
