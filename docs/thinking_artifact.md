# AlphaAgents — Design Thinking Artifact

This document captures the key design choices made during development, the trade-offs
considered, and the reasoning behind each decision. Intended for portfolio review.

---

## Problem Framing

Equity research is expensive and slow. A junior analyst spends 8–12 hours producing a
single research note: gathering data, reading filings, synthesising news, writing
structured output, and getting it reviewed. The hypothesis: an LLM pipeline can produce
a first-draft note in under 3 minutes that a senior analyst can review and approve in
10 minutes, cutting total time by 70%.

The challenge is trust. Financial AI that invents numbers or makes uncited claims is
worse than no AI — it erodes trust and can cause harm. So the system design prioritises
honesty (explicit data gaps), auditability (citations, LangFuse traces), and human
control (mandatory HITL).

---

## Key Design Tensions

### 1. Speed vs Quality
Running all four worker agents in parallel (web, financial, news, memory) cuts wall-clock
time by ~50% vs sequential execution. The trade-off: parallel agents can't share
intermediate results. The writer sees all outputs simultaneously, which is actually
better for synthesis — it doesn't need sequential context.

### 2. Automation vs Control
The critic+revision loop automates one quality pass. But we cap revision loops at 2
(configurable) to bound cost and latency. The HITL gate ensures a human always sees the
note before it is used. This is a deliberate choice: the system is designed to *assist*
analysts, not replace them.

### 3. Cost vs Capability
Llama-3.1-8B (free) vs GPT-4 (paid). The free model is less reliable at JSON mode and
has lower reasoning quality. We compensate with: explicit JSON-only system prompts,
Pydantic validation with fallback model, and structured critique (so errors are
detectable). For a demo/prototype, this is the right trade-off.

### 4. Flexibility vs Correctness
Using TypedDict (not Pydantic BaseModel) for ResearchState keeps LangGraph's state
merging working correctly (LangGraph expects dict-like state). All inter-agent data
contracts use Pydantic models as *fields* of the TypedDict. This is the idiomatic
LangGraph pattern.

---

## What I Would Do Differently at Scale

1. **Replace SQLite with PostgreSQL** — concurrent writes from multiple pipeline runs
   would contend on SQLite. PostgreSQL with asyncpg handles this cleanly.

2. **Replace ChromaDB with Pinecone or Weaviate** — for >10K notes, a managed vector DB
   with proper indexing and horizontal scaling becomes necessary.

3. **Stream LLM output to the UI** — currently the UI polls every 3 seconds. With
   Server-Sent Events or WebSockets, the writer output could stream token-by-token.

4. **Fine-tune a small extractor model** — instead of using the 8B model for ticker
   extraction (a simple NER task), a fine-tuned 1B model would be faster and cheaper.

5. **Add a debate agent** — a second LLM takes the opposite investment thesis and
   argues against the recommendation, forcing the writer to address counter-arguments.
