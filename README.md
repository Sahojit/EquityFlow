# AlphaAgents — Agentic Equity Research Analyst

> A multi-agent LLM pipeline that produces cited, structured investment research notes
> in under 3 minutes, with a human-in-the-loop approval gate before any note is published.

---

## Demo

> **TODO:** Record a 3–5 min Loom walkthrough and paste the embed link here.  
> **Live URL:** TODO — add Render deployment URLs after first deploy.

---

## Problem Statement

Equity research analysts spend 8–12 hours producing a single research note: gathering
web data, pulling financial metrics, reading recent news, synthesising everything into a
structured document, and getting it peer-reviewed. AlphaAgents compresses the first-draft
stage to under 3 minutes using 6 coordinated LLM agents, a critic review pass, and a
mandatory human-approval gate — so analysts spend their time on judgement, not data
gathering.

---

## Architecture

![Architecture Diagram](docs/architecture.png)

See [docs/architecture.md](docs/architecture.md) for the full C4 Level 2 narrative.

**Pipeline topology:**

```
Query → Orchestrator → [Web Researcher | Financial Data | News | Memory] (parallel)
      → Writer → Critic → Revision Router → (loop or) HITL → Done
```

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Agent framework | LangGraph (StateGraph) | Explicit state transitions, parallel fan-out, conditional revision loop, testable nodes |
| LLM provider | HuggingFace Inference API (free) | $0 cost; OpenAI-compatible; Llama-3.1-8B as primary |
| Web / news search | Tavily API | Structured JSON results, news topic filter, free tier |
| Financial data | yfinance | No API key, covers major global exchanges |
| Vector memory | ChromaDB + sentence-transformers | Local persistence, zero infra cost |
| API layer | FastAPI | Async-native, automatic OpenAPI docs, background tasks |
| Frontend | Streamlit | Rapid UI with polling, tabs, markdown rendering |
| Observability | LangFuse | Per-node spans, trace grouping, error tagging |
| Evaluation | ragas + custom citation_precision | Factuality scoring + citation density metric |
| Storage | SQLite + aiosqlite | Zero-infra job tracking and HITL persistence |
| Linting | ruff | Fast, opinionated, replaces flake8 + isort |
| Type checking | mypy (strict) | Catches type errors before runtime |
| Testing | pytest + unittest.mock | All external APIs mocked; no test doubles touch real endpoints |
| CI | GitHub Actions | Lint + typecheck + test on every push to main |
| Deployment | Render free tier | FastAPI + Streamlit as two separate web services |
| Package manager | uv | Fast, lockfile-based, drop-in pip replacement |

---

## Quickstart

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A free [HuggingFace](https://huggingface.co/settings/tokens) token
- A free [Tavily](https://app.tavily.com) API key
- A free [LangFuse](https://cloud.langfuse.com) account (public + secret key)

### Install

```bash
git clone <your-repo-url>
cd alpha-agents
cp .env.example .env
# Fill in your HF_TOKEN, TAVILY_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
uv sync --all-extras
```

### Run

```bash
# Terminal 1 — FastAPI backend
make run-api

# Terminal 2 — Streamlit frontend
make run-ui
```

Open `http://localhost:8501` in your browser.

### Test

```bash
make test
# or individually:
make lint
make typecheck
```

---

## Data

See [docs/data.md](docs/data.md) for schemas, sources, and field-level documentation
for all data flowing through the pipeline.

---

## Architecture Decision Records

| ADR | Decision |
|-----|----------|
| [ADR-001](docs/adr/ADR-001-langgraph-vs-crewai.md) | LangGraph over CrewAI / AutoGen |
| [ADR-002](docs/adr/ADR-002-memory-architecture.md) | ChromaDB + local embeddings |
| [ADR-003](docs/adr/ADR-003-critic-design.md) | LLM-as-judge critic with revision loop |
| [ADR-004](docs/adr/ADR-004-hallucination-mitigation.md) | Four-layer hallucination mitigation |
| [ADR-005](docs/adr/ADR-005-deployment-strategy.md) | HuggingFace + Render free tier |

---

## Known Limitations

- **Cold starts:** Render free tier spins down after 15 min inactivity; first request takes ~30s.
- **yfinance data gaps:** Many international tickers return incomplete or empty data; `data_available=False` is set and the note will flag figures as "data unavailable".
- **Hallucination rate on financials:** Llama-3.1-8B can still generate plausible but false financial figures; the critic pass and HITL gate are mitigations, not guarantees.
- **Small eval set:** 20 hand-curated queries is sufficient for a prototype benchmark but not statistically rigorous.
- **HuggingFace rate limits:** ~10 req/min on free tier; parallel agent calls can hit this; `call_with_backoff()` retries but adds latency.
- **Non-commercial yfinance:** Yahoo Finance data is not licensed for commercial use; this system is for research/educational purposes only.

---

## Roadmap

- [ ] Debate agent — second LLM argues the bear case to stress-test the recommendation
- [ ] Streaming UI — token-by-token writer output via Server-Sent Events
- [ ] Real-time market data — replace yfinance with a paid data provider (Alpha Vantage, Polygon)
- [ ] Fine-tuned extractor — 1B model for ticker/entity extraction, faster than 8B
- [ ] PostgreSQL + Pinecone — production-grade storage replacing SQLite + ChromaDB

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

Built during the LLM Systems & Applied GenAI internship segment (E1 — Agentic Research Analyst track).  
Powered by: [LangGraph](https://langchain-ai.github.io/langgraph/), [HuggingFace](https://huggingface.co), [Tavily](https://tavily.com), [LangFuse](https://langfuse.com), [FastAPI](https://fastapi.tiangolo.com), [Streamlit](https://streamlit.io).
