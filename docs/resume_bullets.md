# AlphaAgents — Resume Bullets

> Customise these bullets to match the exact job description you are targeting.
> Numbers in [brackets] should be filled in after running the eval suite.

## Software Engineering / ML Engineering Roles

- Built **AlphaAgents**, a production-grade multi-agent equity research system using
  **LangGraph StateGraph**, coordinating 6 specialised LLM agents (orchestrator, web
  researcher, financial data, news, memory, critic) in a parallel + conditional-revision
  topology deployed on Render.

- Engineered a **zero-hallucination contract** across the pipeline: explicit
  `data_available` flag on yfinance failures, mandatory citation enforcement in writer
  prompts, LLM-as-judge critic with structured `CritiqueItem` verdicts, and a
  mandatory HITL approval gate — reducing unsupported claims by an estimated [X]%.

- Implemented a **fault-tolerant LLM client** (`llm/client.py`) with Pydantic-validated
  structured outputs, automatic fallback from Llama-3.1-8B to Mistral-7B on JSON parse
  failure, and exponential backoff on HTTP 429 rate-limit errors.

- Achieved **[X]% citation precision** across 20 eval queries (custom metric: fraction
  of sentences in full_text containing a cited URL), measured via `eval/run_eval.py`.

- Deployed **FastAPI** REST backend and **Streamlit** frontend as separate Render web
  services with GitHub Actions CI (ruff lint + mypy type-check + pytest, all gated on
  `main`).

## Data / AI Research Roles

- Designed a **two-tier memory architecture**: local sentence-transformers embeddings
  (all-MiniLM-L6-v2, no API cost) + ChromaDB persistent vector store, enabling
  contextual retrieval of prior research notes for repeated company queries.

- Integrated **LangFuse** observability with per-node spans and a shared trace ID
  threaded through pipeline state, enabling latency profiling and error triage across
  multi-agent runs.

- Evaluated pipeline factuality using **ragas** and a custom citation precision script
  on 20 hand-curated equity research queries spanning Indian and US large-cap equities.
