# ADR-004: Hallucination Mitigation Strategy

**Status:** Accepted  
**Date:** 2026-06-22

## Context

Financial AI is a high-stakes domain. Hallucinated revenue figures, fabricated deal
wins, or incorrect recommendations can cause real harm if acted upon. yfinance has
data gaps (missing fields for many tickers), and web search results may be inaccurate
or outdated. The LLM can generate plausible-sounding but false financial statements.

## Decision

A four-layer mitigation stack:

1. **Explicit `data_available` flag** — `FinancialData.data_available=False` when yfinance
   returns no usable data. The writer system prompt mandates "data unavailable" text for
   all figures in this case.

2. **Citation mandate in writer prompt** — The writer is instructed to cite every factual
   claim with an inline URL from the provided source list. Claims without a URL are
   explicitly prohibited by the system prompt.

3. **LLM critic pass** — The critic reviews the note for unsupported/missing-citation
   claims and triggers a revision loop if >3 are found (see ADR-003).

4. **Mandatory HITL** — Every note must pass through a human reviewer before being
   marked "approved". The pipeline never auto-approves.

## Consequences

**Positive:**
- Honest notes — uncertain data is labelled, not fabricated
- Multi-layer defence — any single layer failure is caught by the next
- HITL ensures a human sees every note before it is used

**Negative:**
- Slower pipeline — critic pass adds ~10–20s per run
- More user friction — HITL requires manual action
- Does not eliminate hallucinations — mitigates them

## Note on Responsible AI

This system is intended for research and educational use. It is not a licensed
financial advisor. Users should treat all outputs as starting points for further
human research, not as actionable investment advice.
