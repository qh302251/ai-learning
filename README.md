# AI Agent 学习与实践

从零手写 ReAct → LangGraph 重构 → RAG 评估闭环 → 多用户服务化与容器化。

在 **RTX 2060 6GB（0 显存方案：不训练、不做推理优化，走 API + 框架 + 工程）** 的约束下，
走完一条大模型应用工程的完整链路。

> 🎯 **旗舰项目** → [`agent/project/assistant/`](agent/project/assistant/)
> 个人知识库助手：RAG + Agent + FastAPI + Docker

---

## 📊 亮点速览

| 维度 | 结果 |
|---|---|
| **检索质量** | Recall@3 **0.80 → 1.00**，MRR **0.933**（10/10 命中 top-3，9 条排第 1） |
| **生成质量** | correctness **0.82 → 0.92**；faithfulness / answer_relevancy 均 **1.00** |
| **分块改造** | 38 → 31 chunks，最长块 **874 → 390** 字符，**0 处句中切断** |
| **代码收敛** | `agent_core.py` 136 行 → `agent_lg.py` 113 行（**消掉 60 行手写流式解析器**） |
| **首字延迟** | TTFT 615 / 1107 / 1180 ms；端到端 0.93 / 1.74 / 1.74 s |
| **工程质量** | **24 项单元测试全绿** · **9 个工具** · 镜像 **2.05 GB** |
| **生产化** | Bearer 鉴权 / 会话与记忆按用户隔离 / 令牌桶限流 / SSE 流式 / SQLite Checkpointer |

---

## 📁 目录结构

```
.
├── agent/
│   ├── basics/               # 8 步手写 Agent 演进（v1 基础调用 → v8 MCP）
│   ├── micro/                # 手写 ReAct (L0→L2) 与 LangGraph 最小实验
│   ├── rag/                  # RAG 检索实验
│   ├── mcp/                  # MCP 协议实践
│   └── project/assistant/    ★ 旗舰项目：个人知识库助手
├── algorithm/
│   ├── micro/                # 手写算法微实验（LRU / TopK / BM25 / 自注意力 / 多头注意力 …）
│   ├── eval/                 # RAG 评估闭环（M1 检索指标 / M2 生成指标）
│   └── archive/              # 归档
├── llm/
│   ├── inference/            # 推理与量化对比
│   ├── finetune/             # Qwen2.5 LoRA 微调 / 合并 / 对比
│   ├── prompt/               # Prompt 工程实验
│   ├── evaluate/             # 评估脚本
│   └── deploy/               # 部署实验
└── notes/                    # 学习路线 · 面试提词卡 · 错题库
```

---

## 🗺 开发历程

> 首次开源时，Git 历史被压缩为单个提交 —— 目的是彻底移除早期误提交进历史里的密钥
> （当前仓库中已无任何真实凭据）。里程碑保留在此：

| # | 里程碑 | 对应提交 |
|---|---|---|
| 1 | 手写 ReAct (L0→L2) + 8 步 `agent/basics` 演进 | 首次开源前完成 |
| 2 | RAG eval 完成（Recall@3 1.00 / MRR 0.933）+ BM25 集成 + 句子级分块 | `snapshot:` |
| 3 | `algorithm/` 整理为 `eval` / `micro` / `archive` 三层 | `refactor:` |
| 4 | `.env` 隔离密钥 + `.gitignore` 排除敏感配置 | `security:` |
| 5 | 17 个文件改用 `common_config` 读取密钥 | `security:` |
| 6 | 路线图更新 + 真 BM25 + 分块改造 | `docs:` |
| 7 | MCP 跨进程编码健壮性（显式 UTF-8，不依赖系统默认 GBK） | `fix:` |
| 8 | `eval/` 并入项目 + 文件相对定位 + README；M2 correctness 0.82→0.92 | `feat:` |
| 9 | 修复过期的 `rag_tools` 测试；24 项测试全绿 | `test:` |
| 10 | 多用户生产化：Bearer 鉴权 / 会话与记忆隔离 / 令牌桶限流 / SQLite 并发 | 首次开源一并提交 |

---

## 🛠 技术栈

| 层 | 选型 |
|---|---|
| Agent 编排 | LangGraph 1.2 · LangChain 1.4 · 手写 ReAct 打底 |
| 模型接入 | DeepSeek API（OpenAI 兼容）· Qwen2.5 1.5B/7B 本地部署 + LoRA + 4bit 量化 |
| 检索 | 句子级分块 · BM25 + 向量混合检索 · jieba · sentence-transformers |
| 评估 | 自建 golden 集 · Recall@k / MRR · 手写 LLM-as-Judge · RAGAS |
| 服务化 | FastAPI · SSE 流式 · 令牌桶限流 · SQLite Checkpointer |
| 协议 | MCP（Model Context Protocol） |
| 部署 | Docker · Docker Compose · 多阶段镜像 + 私有源加速 |
| 科学计算 | PINN 求解 PDE · 混合有限元（Matlab） |

---

## 🚀 快速开始

```bash
cd agent/project/assistant
# 详细步骤见该目录下的 README
```

- 配置：把 `.env.example` 复制为 `.env` 并填入自己的 key（`.env` 已被 `.gitignore` 忽略，不会入库）
- 冒烟测试：24 项单元测试，**不需要 API Key**
- 启动：CLI 交互 / FastAPI 服务（带流式前端）/ Docker

👉 **[完整说明见 `agent/project/assistant/README.md`](agent/project/assistant/README.md)** ——
里面有量化结果、架构图、Docker 部署，以及 **8 条真实的工程踩坑记录**。

---

## 🔐 密钥安全

- 所有密钥通过 `.env` 注入，`.env` / `*.env` / `.env.*` 全部在 `.gitignore` 中
- 仓库内**不含任何真实凭据**，仅 `agent/project/assistant/` 内有用于本地演示的固定假 key
- 曾在开发早期误将密钥写入源码，已通过 `common_config` 统一收口并清理历史

---

## 📄 License

[MIT](LICENSE)