"""
core/agents.py
==============
Defines the two LangGraph node functions:

  Agent A (Manager/Planner)
  ─────────────────────────
  • Uses DeepSeek Tool Calling to decide routing (NO brittle string parsing).
  • If `tool_calls` is detected → routes to Agent B for RAG.
  • If no tool call → replies directly and ends.
  • On second invocation (after Agent B provides context) → synthesizes final answer.

  Agent B (Rerank-RAG Specialist)
  ────────────────────────────────
  • Called only when Agent A issues a `search_knowledge_base` tool call.
  • Executes the RAGEngine hybrid pipeline.
  • Returns a ToolMessage with strict context boundary prefix so Agent A
    knows exactly which part of context came from the knowledge base.

Boundary Conditions Handled:
  - Empty RAG results → returns a graceful "no result" ToolMessage.
  - API timeout / network errors → raises with clear error message.
  - Unexpected model output → falls back to direct answer mode.
"""

import logging
from typing import Sequence

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from core.config import get_settings
from core.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tool Definition — This is what Agent A can "call" to trigger RAG
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the enterprise medical & nutrition knowledge base for factual information.
    Use this tool when the user's question requires retrieving specific facts,
    research findings, clinical guidelines, or data from internal documents.
    DO NOT use this for general reasoning, calculations, or casual conversation.

    Args:
        query: A precise, self-contained search query in Chinese or English.
    """
    # This function body is intentionally minimal — the actual execution
    # is intercepted by Agent B node in the LangGraph pipeline.
    return query  # placeholder; never actually called


# ─────────────────────────────────────────────────────────────────────────────
# Shared LLM instance — created once at module load, reused for all requests
# ─────────────────────────────────────────────────────────────────────────────

def _create_llm() -> ChatOpenAI:
    """Build the DeepSeek LLM client (singleton-friendly, no side effects)."""
    cfg = get_settings()
    return ChatOpenAI(
        model=cfg.deepseek_model,
        base_url=cfg.deepseek_base_url,
        api_key=cfg.deepseek_api_key,
        temperature=0,
        timeout=60,         # seconds — boundary: prevents hanging on slow network
        max_retries=2,      # boundary: retry transient network errors automatically
    )


# Module-level singleton LLMs
_base_llm: ChatOpenAI | None = None
_agent_a_llm: ChatOpenAI | None = None  # with tools bound


def get_agent_a_llm() -> ChatOpenAI:
    """Lazy singleton for Agent A LLM with tool binding."""
    global _agent_a_llm
    if _agent_a_llm is None:
        _agent_a_llm = _create_llm().bind_tools([search_knowledge_base])
    return _agent_a_llm


def get_base_llm() -> ChatOpenAI:
    """Lazy singleton for base LLM (no tools — used for Agent B synthesis)."""
    global _base_llm
    if _base_llm is None:
        _base_llm = _create_llm()
    return _base_llm


# ─────────────────────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_A_SYSTEM_PROMPT = """你是一个医疗营养领域的智能专家助手（Agent A），你的核心任务是从企业知识库中检索信息来回答问题。

## 何时调用 search_knowledge_base 工具（以下情况必须调用）
- 用户询问任何医疗、营养学、临床、健康相关的事实或概念（例如：某种营养素的功能、某类疾病的饮食原则、某种食物的营养价值、某项研究的结论）
- 用户询问 HealthGenie 系统、本平台的功能、产品特性、互动系统能做什么
- 用户询问涉及企业文档、规范、指南或论文中的信息
- 用户的问题像是在寻找"答案"而非进行"对话"（如提问句式："是什么"、"怎么做"、"有什么"、"为什么"、"多少"等加上领域词）

## 何时直接回答（不调用工具）
- 纯粹的问候/闲聊（如"你好"、"谢谢"）
- 明确的数学计算（如"帮我算 35 加 78"）
- 与医疗营养领域完全无关的通用问题


## 《最高优先级》收到知识库检索结果后的合成规则
当消息历史中出现以「【知识库检索结果】」开头的 ToolMessage 时，进入合成模式：
1. 只能将 ToolMessage 的内容整理改写，绝对禁止添加任何 ToolMessage 中没有的信息。
2. ToolMessage 包含几条信息，答案就包含几条，不得自行扩充或补全。
3. 如果 ToolMessage 说明知识库无相关内容，直接如实告知用户，不要用自身知识补充。
4. 禁止在答案末尾追加「您想了解更多……」等引导语，直接结束。

## 执行规则
- **不确定是否该调用工具时，默认调用工具检索，不用自身知识回答领域问题。**
- 已有检索结果（ToolMessage）后不再调用工具。
- 回答简洁精炼，直接给出结论，不要寒暄或重复铺垫。"""

_AGENT_B_SYSTEM_PROMPT = """你是 Agent B（知识库检索专家）。你会收到从知识库检索到的原始文本片段。
请压缩提炼这些片段，只保留与问题直接相关的核心内容，用简短清晰的中文输出，不要罗列无关细节，不要重复原文，不要添加文档中没有的内容。
如果检索内容与问题相关性不高，一句话说明即可。"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: apply memory sliding window
# ─────────────────────────────────────────────────────────────────────────────

def _apply_memory_window(messages: Sequence[BaseMessage], window: int) -> list[BaseMessage]:
    """
    Trim the message history to the last `window` conversation turns.
    Each turn = 1 HumanMessage + 1 AIMessage = 2 messages.
    The system message (if any) is always preserved at position 0.

    Args:
        messages: Full message history from AgentState.
        window:   Max number of turns (pairs) to keep.

    Returns:
        Trimmed list with at most (window * 2) recent messages.
    """
    # Separate system message from conversation messages
    if messages and isinstance(messages[0], SystemMessage):
        sys_msg = [messages[0]]
        conv_msgs = list(messages[1:])
    else:
        sys_msg = []
        conv_msgs = list(messages)

    # Keep only the last N turns
    max_msgs = window * 2
    if len(conv_msgs) > max_msgs:
        conv_msgs = conv_msgs[-max_msgs:]
        logger.debug(f"Memory pruned: keeping last {window} turns ({max_msgs} messages).")

    return sys_msg + conv_msgs


# ─────────────────────────────────────────────────────────────────────────────
# Node: Agent A
# ─────────────────────────────────────────────────────────────────────────────

def agent_a_node(state: dict) -> dict:
    """
    Agent A node function for LangGraph.

    Inputs (from state):
        messages (Sequence[BaseMessage]): Full conversation history.

    Outputs (state updates):
        messages: Appends the new AIMessage from Agent A.

    Routing signal:
        If returned AIMessage has tool_calls → LangGraph conditional edge routes to agent_b.
        Otherwise → END.
    """
    cfg = get_settings()
    messages = state["messages"]

    # Apply sliding window memory pruning
    windowed_messages = _apply_memory_window(messages, window=cfg.memory_window)

    # Prepend system prompt
    input_messages = [SystemMessage(content=_AGENT_A_SYSTEM_PROMPT)] + windowed_messages

    logger.info(
        f"Agent A invoked | messages_total={len(messages)} | "
        f"messages_after_prune={len(windowed_messages)}"
    )

    try:
        response: AIMessage = get_agent_a_llm().invoke(input_messages)
    except Exception as e:
        logger.error(f"Agent A LLM call failed: {e}", exc_info=True)
        # Boundary: on API failure, return a graceful error message
        fallback = AIMessage(content="抱歉，目前与专家系统的连接出现问题，请稍后重试。")
        return {"messages": [fallback]}

    logger.info(
        f"Agent A response | has_tool_calls={bool(response.tool_calls)} | "
        f"content_preview={str(response.content)[:80]}"
    )
    return {"messages": [response]}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Agent B (RAG Specialist)
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_b_node(rag_engine: RAGEngine):
    """
    Factory that creates the Agent B node with the pre-loaded RAGEngine injected.
    Called once during startup to bind the model to the node.

    Args:
        rag_engine: The pre-built RAGEngine instance (from lifespan).

    Returns:
        A callable node function compatible with LangGraph's StateGraph.
    """

    def agent_b_node(state: dict) -> dict:
        """
        Agent B node function.

        Inputs (from state):
            messages (Sequence[BaseMessage]): Must include Agent A's AIMessage
                                              with tool_calls as the last message.

        Outputs (state updates):
            messages: Appends a ToolMessage with the RAG result (with boundary prefix).
        """
        messages = state["messages"]

        # Extract the tool call from Agent A's last message
        last_ai_message: AIMessage = messages[-1]
        if not last_ai_message.tool_calls:
            logger.warning("Agent B invoked but last AIMessage has no tool_calls. Skipping.")
            return {"messages": []}

        tool_call = last_ai_message.tool_calls[0]
        query = tool_call.get("args", {}).get("query", "").strip()
        tool_call_id = tool_call.get("id", "tool_call_0")

        logger.info(f"Agent B invoked | query='{query}' | tool_call_id={tool_call_id}")

        if not query:
            # Boundary: empty query extracted from tool call
            logger.warning("Agent B: empty query extracted from tool_call args.")
            rag_result = "【知识库检索结果】：检索词为空，无法执行检索。"
        else:
            try:
                raw_result = rag_engine.query(query)
                # Boundary: distinguish empty vs valid results
                if not raw_result.strip() or raw_result.strip().lower() in ("none", "empty response"):
                    logger.info(f"Agent B: RAG returned empty result for query='{query}'")
                    rag_result = f"【知识库检索结果】：在知识库中未找到与 '{query}' 高度相关的内容，建议换个角度重新提问。"
                else:
                    # Agent B uses DeepSeek to summarize/denoise the raw RAG output
                    synthesis_messages = [
                        SystemMessage(content=_AGENT_B_SYSTEM_PROMPT),
                        HumanMessage(
                            content=f"用户的检索意图：{query}\n\n知识库原始检索内容：\n{raw_result}"
                        ),
                    ]
                    b_response = get_base_llm().invoke(synthesis_messages)
                    # Strict context boundary prefix — prevents hallucination bleed
                    rag_result = f"【知识库检索结果】\n{b_response.content}"

            except RuntimeError as e:
                logger.error(f"Agent B: RAGEngine.query failed: {e}", exc_info=True)
                rag_result = f"【知识库检索结果】：知识库检索过程中发生错误，请稍后重试。（{e}）"
            except Exception as e:
                logger.error(f"Agent B: unexpected error: {e}", exc_info=True)
                rag_result = "【知识库检索结果】：知识库服务暂时不可用，请稍后重试。"

        # Return as ToolMessage — this is what LangGraph expects after a tool call
        tool_message = ToolMessage(
            content=rag_result,
            tool_call_id=tool_call_id,
        )
        logger.info(f"Agent B completed | result_preview={rag_result[:100]}")
        return {"messages": [tool_message]}

    return agent_b_node
