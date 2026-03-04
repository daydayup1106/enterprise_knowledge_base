# 系统流转逻辑图描述 (System Flow Logic Diagram)

本系统采用有向无环图 (LangGraph) 实现复杂业务的工作流转，底层依托 LlamaIndex 提供 RAG 数据召回支持。

## 1. 节点与角色定义
- **入口 (Input)**: FastAPI 接收 HTTP 形式的用户提问。
- **Agent A 节点 (Manager)**: 由 `gpt-4o` 驱动的总控大脑。
- **Agent B 节点 (RAG Specialist)**: 由 `gpt-3.5-turbo` 驱动的检索专家。
- **Tool 节点 (LlamaIndex Retriever)**: Agent B 所专属调用的知识库引擎。

## 2. 状态结构 (State Schema)
在 LangGraph 中流转的核心状态包含：
- `messages`: List[BaseMessage]，存储所有的对话历史和函数调用记录。
- `current_agent`: 当前处理事务的代理人标识。

## 3. 流转逻辑步骤
1. **用户提问阶段**: 
   - 用户访问系统，发送 Message `"公司的年假政策是什么？"`。
   - 请求进入 LangGraph，触发初始节点，路由至 **Agent A**。

2. **意图判断阶段 (Agent A)**:
   - Agent A (gpt-4o) 会审视问题。它被赋予了系统提示词：“如果你被问及关于公司事实/规章等特定信息，将流程转交/请求给 Agent B 进行检索。”
   - Agent A 决定需要查询知识库，触发边界路由 (Conditional Edge)，流程流向 **Agent B**。
   *(如果问题是“1+1等于几”，Agent A 将直接回答，流程结束，流向 END).*

3. **专精检索阶段 (Agent B + LlamaIndex)**:
   - Agent B 接收到 Agent A 传递的搜索意图，开始执行其绑定的 LlamaIndex 查询工具 (Tool)。
   - 工具调用：`query_engine.query("公司的年假政策是什么")`。
   - **LlamaIndex 内部处理**: 首先根据前序利用 `SimpleDirectoryReader` 录制在向量库（如通过 Chroma/PGVector 或简单内存字典建立的本地索引）计算 Embeddings 的 Top-K 节点。然后利用大模型（或 Reranker 模型）对结果进行整理。

4. **总结归纳反馈阶段**:
   - `LlamaIndex` 向 Agent B 吐出包含事实知识的纯净段落摘要。
   - Agent B 根据召回的信息组装为回答，存入 state 的 `messages`。
   - 流程流回 **Agent A**（或者根据设计，Agent B 直接结束当前轮次对话流向 END）。
   - 若流回 Agent A，Agent A 会综合润色后输出最终结果。

5. **持久化记录**:
   - 在图 (Graph) 的每个步骤节点，`RedisSaver` (Checkpointer) 会将图切片的状态存储进 Redis（基于 `thread_id`），从而实现多轮长对话与记忆持久化。
