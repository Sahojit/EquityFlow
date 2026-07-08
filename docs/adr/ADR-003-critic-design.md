# ADR-003: Critic Agent Design — LLM-as-Judge with Revision Loop

**Status:** Accepted  
**Date:** 2026-06-22

## Context

Financial research notes can contain hallucinated figures, unsupported claims, and
missing citations. We need a quality gate before the note reaches the human reviewer.
The gate must be automated (no human per-claim review), structured (machine-readable
verdicts), and bounded (finite revision passes to avoid infinite loops).

## Decision

An **LLM-as-judge critic agent** that:
1. Reviews the draft note and outputs a `list[CritiqueItem]` (each item: claim, verdict, reason)
2. Verdicts: `"supported"`, `"unsupported"`, `"missing_citation"`
3. Feeds into a conditional revision router: if >3 unresolved items AND revision_count < MAX_REVISION_LOOPS → route back to writer; else → HITL
4. MAX_REVISION_LOOPS = 2 (configurable via env var)

## Consequences

**Positive:**
- Catches obvious unsupported claims without rule-based citation parsing
- Structured output allows the revision router to make a deterministic decision
- Bounded loop (MAX_REVISION_LOOPS) prevents runaway costs
- Adds ~1 LLM call (~$0.01–0.03 on paid APIs) per pass — negligible on HuggingFace free tier

**Negative:**
- LLM judge can itself hallucinate verdicts (meta-hallucination)
- Does not catch numerical errors — only citation presence
- A sophisticated adversarial note can trick the judge into "supported" verdicts

## Alternatives Considered

| Option | Reason rejected |
|--------|----------------|
| **Rule-based citation checker** | Too brittle — can't distinguish a claim that needs citation from one that is common knowledge |
| **Human-only review** | Defeats the purpose of the automated pipeline; HITL is retained as final gate |
| **RAG-based fact-check** | Would require ground-truth financial databases; out of scope for v1 |
