# AI Agent 开发学习计划

## 当前进度

- [x] ReAct 循环原理
- [x] Tool Use / Function Calling
- [x] Agent Loop 手写（50行最小Agent）
- [x] 给Agent加自定义工具（read_file, write_file）
- [x] 连续对话
- [x] 流式输出（Streaming）
- [x] System Prompt 设计
- [x] 错误处理与重试（网络异常、工具异常、HTTP 错误码捕获）
- [x] RAG（检索增强生成）
  - [x] 关键词搜索工具（search_documents + knowledge 目录）
  - [x] Embedding 与向量搜索（语义匹配，解决"差一个字就搜不到"）
  - [x] 将向量搜索接入 Agent（semantic_search 注册为工具）
- [x] 记忆系统
  - [x] 长期记忆存储（remember + memory.json 持久化）
  - [x] 记忆检索（recall + 关键词匹配）
  - [x] 启动时记忆注入（加载记忆到 system prompt）
  - [x] 全局缓存优化（避免重复读文件）
- [x] MCP 协议
  - [x] MCP Server 实现（stdin/stdout JSON-RPC）
  - [x] MCP Client 集成（工具发现 + 远程调用）
  - [x] Agent Loop 中本地/MCP 工具统一调度
- [x] 项目实战 — 个人知识库助手（模块化架构）
  - [x] 模块拆分：agent_core / tool_registry / 独立 tools 包
  - [x] API Key 通过环境变量注入
  - [x] HF 镜像源配置解决模型下载问题
  - [x] 自动建目录（write_file + os.makedirs）
  - [x] 新增 delete_file 工具
  - [x] 完整的对话测试覆盖（文件操作、RAG、记忆、MCP）
- [ ] 评估体系
  - [x] 单元测试框架搭建（unittest + tests/ 目录）
  - [x] 文件工具测试（read_file / write_file / delete_file）
  - [x] 记忆工具测试（remember / recall，mock 隔离真实数据）
  - [x] ToolRegistry 测试（注册 / 调度 / MCP 合并）
  - [x] Agent 核心循环测试（mock API）
  - [x] RAG 工具测试
- [ ] 面试准备

## 优先级说明

必学 -> RAG / MCP / 错误处理
重要 -> 记忆系统 / 评估体系
可延后 -> Multi-Agent / LangChain / 成本优化

## 学习建议

剩余时间预计15-20小时。调整了两个地方：
- 错误处理提前：写任何代码都会遇到，先学了再做后续实验更顺畅
- 评估体系放到项目实战之后：没有项目，评估无从谈起
