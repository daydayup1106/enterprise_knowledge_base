import os
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 导入 LlamaIndex 相关组件
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- 1. 状态定义 (State Schema) ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    routing_decision: str

# --- 2. 知识库核心集成 (LlamaIndex) ---
def setup_llama_index():
    """
    构建基于 LlamaIndex 的知识库检索引擎，绑定给 Agent B 的轻量级模型。
    """
    Settings.llm = LlamaOpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding()
    
    doc_path = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(doc_path):
        os.makedirs(doc_path)
    
    # 初始化一个 dummy 文件供测试
    if not os.listdir(doc_path):
        with open(os.path.join(doc_path, "company_manual.txt"), "w", encoding="utf-8") as f:
            f.write("《企业知识库管理规范》\n1. 休假制度：满1年员工享受5天年假。\n2. 报销制度：所有餐饮报销需提供发票。\n3. 技术栈要求：必须使用LangGraph和LlamaIndex。")
            
    # SimpleDirectoryReader 读取 /data 目录下的文件
    documents = SimpleDirectoryReader(doc_path).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # 取 top_k=2 进行初步重排检索
    return index.as_query_engine(similarity_top_k=2)

query_engine = setup_llama_index()

# --- 3. 智能体定义 (Agent Nodes) ---
# Agent A 强制使用 gpt-4o
agent_a_llm = ChatOpenAI(model="gpt-4o", temperature=0)
# Agent B 强制使用 gpt-3.5-turbo
agent_b_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def agent_a_node(state: AgentState):
    """
    Agent A (Manager): 意图识别与直接推理节点
    """
    messages = state["messages"]
    last_user_message = [m for m in messages if isinstance(m, HumanMessage)][-1].content
    
    sys_prompt = SystemMessage(content=(
        "你是 Agent A (Manager)。请分析用户的请求：\n"
        "1. 如果用户需要查询具体的企业规章制度、指南或知识点，请直接输出 '<ROUTE_TO_B>' 加上你需要检索的关键词。\n"
        "2. 如果是一般的聊天、计算或者逻辑推理，请直接使用你自己的知识回答，绝对不要包含 '<ROUTE_TO_B>'。"
    ))
    
    response = agent_a_llm.invoke([sys_prompt, HumanMessage(content=last_user_message)])
    content = response.content
    
    if "<ROUTE_TO_B>" in content:
        query_kw = content.replace("<ROUTE_TO_B>", "").strip()
        # 将决策标记存入 state 以供边(Edge)使用
        return {"routing_decision": "to_b", "messages": [AIMessage(content=f"[Agent A Decision] Needs RAG routing: {query_kw}")]}
    else:
        return {"routing_decision": "end", "messages": [AIMessage(content=f"[Agent A] {content}")]}

def agent_b_node(state: AgentState):
    """
    Agent B (Rerank-RAG Specialist): 调用 LlamaIndex 检索并总结
    """
    messages = state["messages"]
    # 提取 Agent A 传递的检索关键词
    last_aimessage = messages[-1].content
    query_kw = last_aimessage.split("Needs RAG routing:")[-1].strip()
    
    # 1. 向 LlamaIndex 发起检索
    retrieval_response = query_engine.query(query_kw)
    context_str = str(retrieval_response)
    
    # 2. Agent B 利用 gpt-3.5-turbo 对知识进行去噪和润色总结
    sys_prompt = SystemMessage(content="你是 Agent B (知识库检索专家)。请基于以下检索到的上下文，简明扼要地回答用户的问题。如果上下文不相关，请如实告知。")
    response = agent_b_llm.invoke([
        sys_prompt, 
        HumanMessage(content=f"上下文: {context_str}\n搜索意图: {query_kw}")
    ])
    
    output_message = AIMessage(content=f"[Agent B (RAG)] {response.content}")
    return {"routing_decision": "end", "messages": [output_message]}

# --- 4. 路由逻辑与流程图编译 (LangGraph Workflow) ---
def should_route(state: AgentState):
    if state.get("routing_decision") == "to_b":
        return "agent_b"
    return END

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent_a", agent_a_node)
workflow.add_node("agent_b", agent_b_node)

# 设置边关联
workflow.set_entry_point("agent_a")
workflow.add_conditional_edges("agent_a", should_route, {"agent_b": "agent_b", END: END})
# Agent B 处理完直接流转到结束
workflow.add_edge("agent_b", END)

# 设置 MemorySaver 用于快速演示时的持久化（生产环境可切换为 RedisSaver）
memory_saver = MemorySaver()
app_graph = workflow.compile(checkpointer=memory_saver)
