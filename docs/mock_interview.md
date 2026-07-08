# AlphaAgents — Mock Interview Q&A

These are the questions most likely to be asked about this project in technical
interviews. Study these before your interview.

---

## Architecture Questions

**Q: Why LangGraph instead of a simple sequential chain?**

> LangGraph gives me explicit state transitions and conditional edges. The revision
> loop — where the critic can send the writer back for a second pass — is a conditional
> edge in the graph. With a chain, I would have had to hand-roll that logic. I also get
> parallel fan-out for free: the four worker agents (web, financial, news, memory) all
> run as parallel edges from the orchestrator node, which cuts wall-clock time roughly
> in half. And every node is a plain Python function, so unit testing is just mocking
> the LLM client.

**Q: How do you handle hallucinations?**

> Four layers. First, if yfinance returns no data, I set `data_available=False` and
> the writer prompt explicitly says to write "data unavailable" rather than invent a
> number. Second, the writer is instructed to cite every factual claim with an inline
> URL from the source list. Third, the critic agent reviews the note and flags
> `unsupported` or `missing_citation` claims — if more than 3 are found, it loops back
> to the writer for revision. Fourth, every note requires human approval before it is
> marked done. No note is auto-approved.

**Q: What happens if the HuggingFace API is rate-limited?**

> `call_with_backoff()` in `llm/client.py` catches `RateLimitError` (HTTP 429) and
> retries with exponential backoff: 2s, 4s, 8s, then raises. If the primary model
> returns invalid JSON, `call_structured()` automatically retries once with the fallback
> model before raising `RuntimeError`. So there are two independent retry mechanisms:
> one for rate limits, one for output quality.

---

## Design Decision Questions

**Q: Why HuggingFace free tier instead of OpenAI?**

> $0 constraint. The HuggingFace Inference API is OpenAI-compatible — I just point the
> OpenAI Python SDK at `https://api-inference.huggingface.co/v1/` and swap the API key.
> The trade-off is Llama-3.1-8B is less reliable at JSON mode than GPT-4, which is why
> I have the Pydantic validation + fallback model in `call_structured()`.

**Q: Why not use try/except pass anywhere?**

> Silent exception swallowing is how bugs become invisible. Every exception is either
> logged and re-raised (for fatal errors like missing API keys) or logged and converted
> to a fallback (like `web_results=[]` when Tavily fails). The pipeline always knows
> what happened. The LangFuse spans also record error-level events, so I can see exactly
> which node failed and why.

---

## Technical Deep-dive Questions

**Q: Walk me through what happens when a user submits a query.**

> 1. POST /research → FastAPI creates a UUID job, persists `status=running` to SQLite,
>    launches `_run_pipeline` as a background task, returns the job_id immediately.
> 2. Background: LangGraph orchestrator decomposes the query into 3–5 sub-questions.
> 3. Four worker agents run in parallel: Tavily web search + LLM summary, yfinance
>    fetch, Tavily news + LLM sentiment, ChromaDB memory retrieval.
> 4. Writer synthesises all outputs into a ResearchNote JSON.
> 5. Critic reviews for unsupported claims. If >3 found and under revision cap, loops
>    to writer; otherwise advances.
> 6. HITL node sets `final_note=draft_note` and the pipeline ends.
> 7. FastAPI updates SQLite to `status=awaiting_hitl`.
> 8. UI poll detects `awaiting_hitl`, routes analyst to Review tab.
> 9. Analyst approves → POST /research/{id}/hitl → SQLite updated to `done`,
>    note stored in ChromaDB.

**Q: What is your citation precision metric and how do you compute it?**

> It's the fraction of non-heading lines in `full_text` that contain at least one URL
> from the `citations` list. I split the text by newline, exclude blank lines and
> markdown headings (lines starting with `#`), and for each remaining line check if any
> citation URL appears in it. The result is a float in [0, 1]. A score of 1.0 means
> every content line has a citation; 0.5 means half do. It's a rough proxy for citation
> density, not factual accuracy.
