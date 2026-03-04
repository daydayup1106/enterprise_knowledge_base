"""
core/graph.py
=============
LangGraph multi-agent state machine.

Flow:
  Entry ──► agent_a ──► [conditional routing]
                            │
                            ├─ tool_calls present? ──► agent_b ──► agent_a (synthesis)
                            │
                            └─ no tool_calls? ──► END

State Schema:
  messages: Annotated[Sequence[BaseMessage], operator.add]
    - Uses operator.add reducer: each node appends new messages.
    - Contains the full conversation (pruning happens inside agent_a_node).
    - Common context shared by both agents.

Persistence:
  - RedisSaver checkpoints state after every node execution.
  - thread_id config key enables per-session isolation.
  - On restart, LangGraph automatically restores from the last checkpoint.
"""

import logging
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
import redis

from core.rag_engine import RAGEngine
from core.agents import agent_a_node, build_agent_b_node
from core.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    LangGraph state shared across all nodes in the graph.

    messages:
        Append-only sequence of BaseMessage objects.
        operator.add ensures new messages from each node are concatenated,
        never overwritten. This maintains full conversation history.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ─────────────────────────────────────────────────────────────────────────────
# Routing Logic
# ─────────────────────────────────────────────────────────────────────────────

def _should_call_rag(state: AgentState) -> str:
    """
    Conditional edge function — determines the next node after agent_a.

    Strategy: Inspect AIMessage.tool_calls (not string parsing).
      • If Agent A issued a tool call → route to "agent_b"
      • Otherwise → END (direct answer, no RAG needed)

    Returns:
        "agent_b" | "__end__"
    """
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.debug(f"Routing to agent_b | tool_call={last_message.tool_calls[0].get('name')}")
        return "agent_b"

    logger.debug("Routing to END (no tool call — direct answer)")
    return END


# ─────────────────────────────────────────────────────────────────────────────
# Graph Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(rag_engine: RAGEngine):
    """
    Build and compile the LangGraph StateGraph.
    Called once at startup with the pre-loaded RAGEngine.

    The graph uses RedisSaver for persistent checkpointing:
    - State is saved after each node execution.
    - On repeated invocations with the same thread_id, LangGraph
      restores the previous state automatically.

    Args:
        rag_engine: Pre-warmed RAGEngine from startup lifespan.

    Returns:
        Compiled LangGraph app (CompiledStateGraph).

    Raises:
        ConnectionError: If Redis is unreachable at startup.
    """
    settings = get_settings()

    # ── Redis Checkpointer ────────────────────────────────────────────────────
    logger.info(f"Connecting to Redis at: {settings.redis_url}")
    try:
        # Use a temporary client only to verify connectivity at startup
        _test_client = redis.from_url(settings.redis_url, socket_connect_timeout=5)
        _test_client.ping()  # fail fast if Redis is down
        _test_client.close()
        logger.info("Redis connection verified.")
    except Exception as e:
        raise ConnectionError(
            f"Cannot connect to Redis at '{settings.redis_url}'. "
            f"Please ensure Redis is running. Error: {e}"
        ) from e

    # RedisSaver expects a URL string, NOT a redis.Redis instance
    checkpointer = RedisSaver(settings.redis_url)
    # Create required Redis search indexes on first run (idempotent — safe to call every startup)
    checkpointer.setup()
    logger.info("RedisSaver indexes initialized.")

    # ── Node Functions ─────────────────────────────────────────────────────────
    agent_b_node = build_agent_b_node(rag_engine)

    # ── Graph Definition ───────────────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    workflow.add_node("agent_a", agent_a_node)
    workflow.add_node("agent_b", agent_b_node)

    # Entry point
    workflow.set_entry_point("agent_a")

    # Conditional routing after Agent A
    workflow.add_conditional_edges(
        "agent_a",
        _should_call_rag,
        {
            "agent_b": "agent_b",
            END: END,
        },
    )

    # After Agent B completes → return to Agent A for synthesis
    workflow.add_edge("agent_b", "agent_a")

    # Compile with Redis checkpointer for persistent state
    app = workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph compiled successfully.")
    return app
