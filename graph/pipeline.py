"""LangGraph pipeline skeleton.

Node functions are imported from the agents package. This module builds and
compiles the StateGraph. Agents are wired in after all agent modules exist.
"""

from __future__ import annotations

import logging
import os

from langgraph.graph import END, START, StateGraph

from graph.state import ResearchState

logger = logging.getLogger(__name__)

MAX_REVISION_LOOPS: int = int(os.getenv("MAX_REVISION_LOOPS", "1"))


def revision_router(state: ResearchState) -> str:
    """Decide whether to send the draft back to the writer or forward to HITL.

    Routes to ``writer_node`` if there are more than 3 unsupported/missing-citation
    claims AND the revision cap has not been reached. Otherwise routes to ``hitl_node``.

    Args:
        state: Current pipeline state after the critic node has run.

    Returns:
        Either ``"writer_node"`` or ``"hitl_node"``.
    """
    critique = state.get("critique", [])
    revision_count = state.get("revision_count", 0)

    unresolved = [
        c for c in critique if c.verdict in ("unsupported", "missing_citation")
    ]

    if len(unresolved) > 6 and revision_count < MAX_REVISION_LOOPS:
        logger.info(
            "Routing back to writer. Unresolved claims=%d, revision=%d/%d",
            len(unresolved),
            revision_count,
            MAX_REVISION_LOOPS,
        )
        return "writer_node"

    logger.info(
        "Routing to HITL. Unresolved claims=%d, revision=%d/%d",
        len(unresolved),
        revision_count,
        MAX_REVISION_LOOPS,
    )
    return "hitl_node"


def hitl_node(state: ResearchState) -> ResearchState:
    """Human-in-the-loop boundary node.

    Sets ``final_note`` to the current ``draft_note`` when the pipeline reaches
    this node. The API layer (/research/{job_id}/hitl) overwrites ``hitl_decision``
    and optionally ``final_note`` once the human acts.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with ``final_note`` populated from ``draft_note``.
    """
    logger.info("Pipeline reached HITL boundary. Awaiting human decision.")
    updated: ResearchState = {**state, "final_note": state.get("draft_note")}
    return updated


def build_graph() -> StateGraph:
    """Construct and return the compiled AlphaAgents LangGraph StateGraph.

    Imports agent node functions lazily to avoid circular imports. The graph
    topology is:

    START → orchestrator → [web_researcher, financial_data, news, memory] (parallel)
          → writer → critic → revision_router
          → writer (if revision needed) | hitl_node → END

    Returns:
        A compiled ``StateGraph`` ready to be invoked with a ``ResearchState`` dict.
    """
    from agents.critic import critic_node
    from agents.financial_data import financial_data_node
    from agents.memory import memory_node
    from agents.news import news_node
    from agents.orchestrator import orchestrator_node
    from agents.web_researcher import web_researcher_node
    from agents.writer import writer_node

    graph = StateGraph(ResearchState)

    graph.add_node("orchestrator_node", orchestrator_node)
    graph.add_node("web_researcher_node", web_researcher_node)
    graph.add_node("financial_data_node", financial_data_node)
    graph.add_node("news_node", news_node)
    graph.add_node("memory_node", memory_node)
    graph.add_node("writer_node", writer_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("hitl_node", hitl_node)

    graph.add_edge(START, "orchestrator_node")

    graph.add_edge("orchestrator_node", "web_researcher_node")
    graph.add_edge("orchestrator_node", "financial_data_node")
    graph.add_edge("orchestrator_node", "news_node")
    graph.add_edge("orchestrator_node", "memory_node")

    graph.add_edge("web_researcher_node", "writer_node")
    graph.add_edge("financial_data_node", "writer_node")
    graph.add_edge("news_node", "writer_node")
    graph.add_edge("memory_node", "writer_node")

    graph.add_edge("writer_node", "critic_node")
    graph.add_conditional_edges(
        "critic_node",
        revision_router,
        {"writer_node": "writer_node", "hitl_node": "hitl_node"},
    )

    graph.add_edge("hitl_node", END)

    return graph.compile()


pipeline = build_graph()
