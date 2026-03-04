# 企业知识库专家协同系统逻辑流程图

这是一个全景展示系统请求流转、多智能体交互 (LangGraph) 以及大模型结合本地知识库(RAG)的完整泳道图/流程图。

```mermaid
sequenceDiagram
    participant User as 用户 (前端)
    participant FastAPI as API 网关 (main.py)
    participant AgentA as Agent A (判断总控)
    participant DeepSeek as 大语言模型 API
    participant AgentB as Agent B (知识提炼)
    participant RAGEngine as RAG Engine (检索)
    participant LocalModels as 本地模型 (BGE)
    participant Redis as Redis (历史与状态)

    %% 生命周期启动
    Note over FastAPI,LocalModels: 系统启动阶段 (Lifespan)
    FastAPI->>RAGEngine: build_rag_engine()
    RAGEngine->>LocalModels: 1. 加载 Embedding (cpu/cuda)
    LocalModels-->>RAGEngine: OK
    RAGEngine->>RAGEngine: 2. 读取/切分文档数据
    RAGEngine->>LocalModels: 3. 构建 VectorStore & BM25
    LocalModels-->>RAGEngine: 返回索引
    RAGEngine->>LocalModels: 4. 加载 Reranker
    LocalModels-->>RAGEngine: OK
    RAGEngine-->>FastAPI: RAG 系统就绪 (index_ready=True)

    %% 多轮对话流
    Note over User,Redis: 运行时对话流转

    User->>FastAPI: POST /api/v1/chat {"message": "吃红枣补血吗？"}
    
    FastAPI->>Redis: 读取会话历史记录 (thread_id)
    Redis-->>FastAPI: 返回历史 messages[]
    
    FastAPI->>AgentA: 把新问题拼接入状态机 (LangGraph)
    
    AgentA->>DeepSeek: 发送携带工具定义(tools)的提问
    Note over DeepSeek: 模型根据 System Prompt 评估
    DeepSeek-->>AgentA: 决策：需要查询知识，返回 ToolCall (search_knowledge_base)
    
    %% LangGraph 条件路由
    Note over AgentA,AgentB: LangGraph 根据 ToolCall 路由到 Agent B
    AgentA->>AgentB: 传递检索指令及原贴文

    AgentB->>RAGEngine: query("吃红枣补血吗？")
    
    rect rgb(240, 248, 255)
        Note over RAGEngine,LocalModels: 双路召回与重排序过程
        RAGEngine->>LocalModels: 生成 Query Embedding
        LocalModels-->>RAGEngine: 返回向量
        RAGEngine->>RAGEngine: 向量检索召回 Top K
        RAGEngine->>RAGEngine: BM25 关键词召回 Top K
        RAGEngine->>LocalModels: 交给 Reranker 交叉重排 Top N
        LocalModels-->>RAGEngine: 返回最相关文本片段
    end

    RAGEngine-->>AgentB: 返回未清洗的原始知识片段
    
    %% Agent B 提炼
    AgentB->>DeepSeek: 带着【严苛级】 Prompt 提炼纯净答案
    Note over DeepSeek: 模型严格基于原文生成摘要，绝不发散扩充
    DeepSeek-->>AgentB: 精简后的中文短句
    
    AgentB-->>AgentA: 返回包装好的 ToolMessage (以【知识库检索结果】开头)
    
    %% 终止/合成
    Note over AgentA,DeepSeek: Agent A 接收到事实并做最后包装
    AgentA->>DeepSeek: 结合上下文及 ToolMessage 渲染最后修辞
    DeepSeek-->>AgentA: 最终严谨合规的回答
    
    AgentA->>Redis: 记录整个会话周期节点至检查点
    AgentA-->>FastAPI: 吐出状态机最新节点 AIMessage
    
    FastAPI-->>User: 返回 {"reply": "...", "used_rag": true}
```

### 流程说明摘要
1. **全局调度 (`main.py`)**：通过 FastAPI 暴露接口，并在 `lifespan` 阶段完成了耗时的 Embedding 和 Reranker 模型显存加载，保证了在有请求来临时直接推理，避免延迟陡增。
2. **多智能体博弈 (`core/agents.py`)**：
   - **Agent A (总控节点)** 负责闲聊应对和意图判断。它像人脑的高级皮层，决定是直接用自带常识说“你好”，还是触发去企业知识库查阅的工具调用动作。
   - **Agent B (检索专家)** 负责从知识库的泥沙里淘金。因为直接交给主模型的原始检索段落太杂，它充当清洗员去提纯核心重点。
3. **混合检索管道 (`core/rag_engine.py`)**: RAG 组件并没有完全依赖大模型，而是利用成熟的（向量 + BM25 双路召回）+ `bge-reranker` （交叉注意力重排）结构，保障了命中企业本地数据的召回准确率。
4. **记忆持久化 (Redis)**: LangGraph 通过 `RedisSaver` 挂载在图上，只要带有同一个 `thread_id`，上文的脉络就会自动携带进模型，实现了真正的跨端协同问答。
