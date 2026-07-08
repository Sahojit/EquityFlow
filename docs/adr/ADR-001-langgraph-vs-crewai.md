# ADR-001: LangGraph vs CrewAI for Multi-Agent Orchestration

**Status:** Accepted  
**Date:** 2026-06-22

## Context

AlphaAgents requires a multi-agent framework that supports:
- Parallel execution of 4 independent worker agents (web, financial, news, memory)
- A conditional revision loop (writer → critic → writer, up to N times)
- A mandatory HITL boundary that halts the pipeline for human input
- Testable, inspectable state transitions for debugging and evaluation

## Decision

**LangGraph** (StateGraph API) was chosen over CrewAI and AutoGen.

## Consequences

**Positive:**
- Explicit, typed state (ResearchState TypedDict) makes every transition auditable
- Conditional edges (`add_conditional_edges`) express the revision loop cleanly
- Fan-out/fan-in parallelism is first-class via multiple edges from one node
- Individual nodes are plain Python functions — easy to unit test with mocks
- LangFuse integrates at the node level without any framework-specific adapter

**Negative:**
- More boilerplate than CrewAI for simple sequential pipelines
- Graph compilation step (`graph.compile()`) adds a minor startup cost

## Alternatives Considered

| Framework | Reason rejected |
|-----------|----------------|
| **CrewAI** | Higher abstraction hides state transitions; harder to inspect intermediate outputs; revision loop requires custom callback wiring |
| **AutoGen** | Conversation-based (agents talk to each other via messages); revision loop and HITL are awkward to express; state is implicit in message history |
| **Plain asyncio** | Would require hand-rolling graph topology, retry logic, and state management — reinventing LangGraph |
