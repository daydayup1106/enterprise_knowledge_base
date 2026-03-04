你是一个资深的全栈 AI 架构师，精通 LangGraph、LlamaIndex、FastAPI 以及 Docker。请基于以下业务逻辑和技术要求，编写一个名为“企业知识库专家协同系统”的完整项目代码。

## 核心规则 (Rules)
1. 必须完全理解需求后，再从需求出发进行代码设计，不得随意处理边界情况。
2. 各个接口的输入输出要对应，前后端衔接要适配。
3. 不得不经允许自行修改需求，不得随意自行执行，必须要我统一后再执行。
4. 必须严格按照要求执行，需求设计要彻底，具体到每个接口的输入输出参数，以及每个接口的异常情况处理。
5. 后端默认使用conda的enterprise_knowledge_base环境，python使用3.11
6. 需要的包要校验是否适配python3.11，包依赖之间是否适配，出现问题及时解决，不得隐瞒。
7. 代码处理要灵活，所有可变参数（如模型名、窗口大小、Top-K 数量等）必须通过配置文件或环境变量注入，不得写死在代码逻辑中。
8. 所有安装任务，必须提前整理成完整的执行步骤和可能出现的问题，写在执行计划最前面，由用户自行执行，不得由模型代为执行任何安装命令。


## 业务逻辑
你需要构建一个基于 LangGraph 的多代理 (Multi-Agent) 协同系统，包含两个核心角色：

1. **Agent A (Manager/Planner)**: 
   - 职责：接收用户问题，判断意图。如果问题属于“需要查询知识库/事实”的需求，则将任务路由给 Agent B 执行检索。如果涉及普通的逻辑推理、格式化、通用计算，则自己直接回答。最终合并 Agent B 的结果返回给用户。
   - 内核：强制使用 `deepseek-chat`。

2. **Agent B (Rerank-RAG Specialist)**:
   - 职责：被 Agent A 调用时，负责利用 LlamaIndex 在本地建立的文档库（需支持本地 5页以上 PDF/MD 文档建立的基础向量索引）中检索相关内容。
   - 要求：检索后需进行初步去噪并总结出与问题高度相关的结果段落，传递回 LangGraph 状态机。
   - 内核：强制使用 `deepseek-chat`。

## 技术栈与要求详情
1. **核心模型**: 使用 DeepSeek API 体系 (模型名 `deepseek-chat`)，在 LangChain 和 LlamaIndex 中配置相对应的 Base URL 和 API Key 进行适配。
2. **状态流转与边界条件**: 
   - 使用 LangGraph 定义节点流转逻辑，必须定义清晰的 state schema，包含 `messages` (用于上下文连贯) 和当前的 `routing_decision`。
   - **记忆修剪 (Memory Pruning)**: 必须在 LangGraph 的状态访问逻辑中显式增加**滑动窗口/记忆修剪逻辑**，只保留最近 N 轮的对话历史进入 LLM，防止无限累加导致 Token 超限或脱轨。
   - **容错处理**: 务必处理好边界条件（如输入为空、API 超时、模型输出未按预定格式走等），保证数据流转严丝合缝匹配输入输出参数。
3. **高级知识库检索架构 (LlamaIndex RAG Pipeline)**: 利用 LlamaIndex 构建本地 RAG，解决高精度要求。
   - **智能切分**: 使用 `SentenceSplitter` (如 512 chunk size, 128 overlap) 或 `MarkdownNodeParser` 处理 `data/medical_ai_papers_2024_2025.md`，保证医学上下文在边界不被硬阻断。
   - **混合检索 (Hybrid Search)**: 必须同时构建基于 `OpenAIEmbedding` (或类似)的向量稠密检索和基于词汇的 **BM25 稀疏检索**。融合两者的召回优势（防止医学专有名词在使用纯向量时丢失）。
   - **交叉编码重排 (Cross-Encoder Reranking)**: 初筛池 (Top-K=10) 必须经过 Cross-Encoder 模型（如 `SentenceTransformerRerank` 使用 BAAI/bge-reranker-base 轻量模型或利用 DeepSeek API 的特定方式/轻量本地模型）进行二次打分，仅保留最精准的 Top-3 切片。
   - **上下文隔离**: Agent B 在组装 RAG 结果时，必须在提示词和最终返回的 `AIMessage` 中**严格划定外部知识的边界**，禁止与用户的闲聊记忆发生污染。
4. **准确率保障与评估 (RAGAS)**: 系统需内置或在额外测试脚本中体现基于 `ragas` 框架的评估逻辑准备，针对 `Faithfulness` (忠实度/反幻觉) 和 `Context Precision` (上下文精度) 的评分代码留痕。
5. **前后端适配与接口**: 使用 FastAPI 封装系统入口。提供一个 API `/chat` 进行对话，同时提供一个静态测试页面。前后端需通过 `thread_id` 完美适配多轮对话的上下文连贯传递机制，处理异步交互中的各类网络状态。
6. **部署架构与环境**: 
   - 运行环境必须限定使用 Conda 的 `enterprise_knowledge_base` 环境。
   - 交付 `Dockerfile` 时，必须基于 miniconda 镜像并在容器内创建/激活 `enterprise_knowledge_base` 环境来执行。
   - 继续提供 `docker-compose.yml` 联合 Redis 一键拉起。
   - 必须使用 Redis 容器进行 LangGraph 的 state 检查点持久化 (RedisSaver)。

## 输出要求
- 请直接输出 Python 项目代码，包含清晰的注释。
- 请提供可以一键启动的 docker 环境配置。
- 代码风格要追求 Vibe Coding 的极致美感，结构清晰且具有极高的鲁棒性。
