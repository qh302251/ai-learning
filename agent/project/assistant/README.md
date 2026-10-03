# 个人知识库助手 — RAG + Agent（0 显存方案）

> 一个**手写核心链路**的 RAG + Agent 项目：自己实现 ReAct 循环、BM25、混合检索（RRF）、Cross-Encoder 精排、MCP 跨进程工具协议与长期记忆，并配套一套**可复现的 RAG 评估闭环**。
>
> 硬件约束：RTX 2060 6GB（无云 GPU）→ **全程 0 显存**。LLM 走 DeepSeek API，向量与重排用本地小模型。

---

## 📊 量化结果

### 检索指标（M1 · 10 条 golden 测试集）

| 阶段 | Recall@3 | MRR |
|---|---|---|
| 初版：char 级关键词 + 固定 300 字符切块 | 0.80 | 0.800 |
| **+ 真 BM25（jieba / TF-IDF）+ 句子级分块** | **1.00** | **0.933** |

> 10/10 全部命中前 3，其中 9 条排在第 1 名。

### 生成指标（M2 · 手写 LLM-as-Judge）

| 指标 | 分块改造前 | **分块改造后** | 含义 |
|---|---|---|---|
| faithfulness | 1.00 | 1.00 | 回答是否完全基于召回上下文（不编造） |
| answer_relevancy | 1.00 | 1.00 | 回答是否切题 |
| **correctness** | 0.82 | **0.920** | 与标准答案比，是否完整正确 |

> ⚠️ **关键发现：faithfulness / answer_relevancy 抓不到"答案不完整"。**
> 有一道题（两个 4 节点单元分别由谁提出）漏了 ECQ4 单元，前两项仍给 1.0，**只有 correctness 给了 0.0**。
> → **评估指标必须互补使用，不能只看"有没有编造"。**

### 分块改造的连带收益

| | 改造前 | 改造后 |
|---|---|---|
| chunk 数 | 38 | 31 |
| 最长块 | 874 字符（含期刊元数据碎片） | **390 字符** |
| 句子被拦腰切断 | 几乎每块都有 | **0** |
| PDF 提取噪声字符 | 有（`\ufffe` / `\x02` 等） | **0** |

**只改分块、没动任何检索算法，Recall@3 就从 0.80 干到 1.00，correctness 从 0.82 到 0.92。**

### 工程指标（M3 · 服务化 + 容器化）

**代码量：框架替掉了多少样板**

| 文件 | 行数 | 说明 |
|---|---|---|
| `agent_core.py`（手写 ReAct） | 136 | 含 **60 行手写「流式 tool_calls 分片拼接」解析** |
| **`agent_lg.py`（LangGraph 版）** | **113** | 那 60 行**一行没写** —— `bind_tools` 直接给出 dict 形式的 `args` |
| `server.py`（FastAPI 服务层） | 169 | 5 个路由 + SSE 编码 + 审批状态判别 |

**服务化实测**（本机 RTX 2060 / 15.8 GB RAM，服务跑在容器里）

| 指标 | 值 |
|---|---|
| **首字延迟 TTFT** | **0.6 ~ 1.2 s**（3 次实测：615 / 1107 / 1180 ms） |
| 端到端（问一句答一句） | **0.9 ~ 1.7 s** |
| 流式粒度 | 每轮 **64 ~ 116 个 `text` 帧**（≈ token 级） |
| 单元测试 | **24 个全绿**（`python -m unittest discover tests`） |

> ⚠️ TTFT 的**大头是 LLM API 的网络往返**，不是本机计算 —— 这是「0 显存方案」的必然结果：
> **延迟买不来，只能靠流式体验掩盖**。

**容器化实测**

| 指标 | 值 |
|---|---|
| 镜像大小 | **2.05 GB**（`kb-assistant:latest`） |
| 构建耗时 | **约 6 分钟**（换国内源后；换源前 30 分钟以上没跑完） |
| 容器内模型下载 | **0 次**（挂宿主机 HF 缓存复用） |
| 冷启动（加载向量模型 + reranker + 起 MCP） | **数十秒** → 所以放 `lifespan` 里**只装一次** |
| 依赖 | 16 条 pinned（`requirements.txt`），`torch` 单独走 CPU 源 |

---

## 🏗 架构

> 分四层看：**前端 → HTTP 服务 → LangGraph 状态图 → 工具层**。
> 前两层是"工程"，第三层是"Agent 编排"，第四层是"能力"。

### 一、请求链路

```
浏览器  static/index.html
  fetch('/chat/stream') + response.body.getReader()
  TextDecoder.decode(value, {stream:true})   ← 中文会跨字节切断，必须 stream:true
  split('\n\n') + parts.pop()                ← 留住最后那半截帧
        │
        │  HTTP  POST /chat/stream     （SSE；一帧 = data: {...}\n\n）
        ▼
FastAPI  server.py                            ← lifespan 里装配一次，进程级复用
  GET  /              极简前端（演示打字机）
  GET  /health        健康检查（容器探活用）
  POST /chat          JSON 一问一答（非流式）
  POST /chat/resume   审批答复，从断点继续
  POST /chat/stream   SSE 流式（推荐）
        │
        │  config = {"configurable": {"thread_id": ...}, "recursion_limit": 10}
        ▼
LangGraph  agent_lg.py                        ← 见下方状态图
```

> **为什么装配放在 `lifespan`**：加载向量模型 + reranker + 起 MCP 子进程要几十秒，
> 放请求里每个请求都得等一次。`lifespan` 保证**整个进程只装一次**。

### 二、LangGraph 状态图（`agent_lg.py`）

```
                        START
                          │
                          ▼
                 ┌─────────────────┐
         ┌──────►│       llm       │  llm_node
         │       │ 调模型 + 拼工具 │  · SystemMessage 现拼，不进 State
         │       │  bind_tools     │  · 返回 AIMessage（可能带 tool_calls）
         │       └────────┬────────┘
         │                │
         │       should_continue(state)
         │            ┌────┴─────┐
         │      有 tool_calls   没有
         │            │          │
         │            ▼          ▼
         │      ┌──────────────┐  END
         │      │    tools     │  tools_node
         │      │ 逐个执行工具 │  · needs_approval? → interrupt() 暂停等人工
         │      └──────┬───────┘  · registry.dispatch(name, **args)
         │             │          · 结果包成 ToolMessage（tool_call_id 对齐）
         └─────────────┘
          执行完回到 llm → 这就是 ReAct 循环
```

**图的三要素**（面试高频）：

| 概念 | 在本项目里是什么 | 关键点 |
|---|---|---|
| **State** | `TypedDict`，只有 `messages` 一个字段 | 定义"图在流转什么数据" |
| **Reducer** | `Annotated[list, add_messages]` | 节点返回**增量**，框架负责**追加 / 按 id 覆盖**，不是整表替换 |
| **Checkpointer** | `SqliteSaver(conn)` | 每个节点执行后落盘快照 → 支持 **暂停 / 恢复 / 时间旅行** |

**`interrupt()` 的关键机制**：它靠**抛异常**暂停，恢复时会**从节点第一行重跑** →
所以节点必须"**可重放**"，审批要放在**所有工具执行之前**（本项目正是这么排的）。

### 三、事件流（`stream_turn` 生成器）

`app.stream(..., stream_mode=["messages", "updates"])` 同时吃两种流：

| 事件 | 来源 | 内容 |
|---|---|---|
| `("text", str)` | `messages` 流 + `meta["langgraph_node"]=="llm"` | token 级文字（**必须过滤节点**，否则 ToolMessage 全文会刷屏） |
| `("tool_call", str)` | `updates` 流的 `llm` 键 | `hybrid_search({'query': ...})` |
| `("tool_done", str)` | `updates` 流的 `tools` 键 | 工具执行完成 / 被人拒绝 |
| `("interrupt", str)` | `updates` 里的 `"__interrupt__"` | 审批问题 |
| `("end", None)` | 生成器末尾 | 本轮结束 |

`server.py` 把每个事件 `json.dumps` 转义后包成 `data: {...}\n\n` 发出去。
**内容必须转义** —— 裸换行会**撕裂 SSE 帧结构**。

### 四、工具层

**ToolRegistry**：统一 schema + 分发。危险工具在 **registry** 里标 `require_approval=True`，
新增危险工具**不用改 Agent 代码**（`main.py` / `agent_lg.py` 都不动）。

| 分类 | 工具 | 说明 |
|---|---|---|
| 本地文件 | `read_file` / `write_file` / `delete_file` | 危险操作标 `require_approval` |
| 长期记忆 | `remember` / `recall` | JSON 持久化到 `memory.json` |
| **RAG** | `search_documents` / **`hybrid_search` ★** | 混合检索（见下） |
| MCP 远程 | `calculator` / `get_weather` | **跨进程** JSON-RPC over stdio |

**`hybrid_search` 内部流水线：**

```
query
  ├─► BM25 关键词路（jieba 分词 + TF/IDF + 长度归一化） ──┐
  └─► 向量语义路（多语言 MiniLM，中英互通） ──────────────┤
                                                          ▼
                                          RRF 融合 → top_k=20 召回池
                                                          │
                                                          ▼
                            Cross-Encoder 精排（bge-reranker-v2-m3）
                                                          │
                                                          ▼
                                            top_n=5 上下文 → 交给 llm
```

---

## 📁 目录结构

```
assistant/
├── main.py                  入口：装配工具 + 交互循环
├── agent_core.py            手写 ReAct Agent 循环（LLM 调用 + 工具调度）
├── tool_registry.py         工具注册表（OpenAI function-calling schema）
├── build_index.py           构建向量索引（切块 → 编码 → embeddings.json）
├── mcp_client.py            MCP 客户端（子进程 + JSON-RPC over stdio）
├── mcp_server.py            MCP 服务端（calculator / get_weather）
│
├── ★ server.py              FastAPI 服务（/health /chat /chat/stream /chat/resume）
├── ★ agent_lg.py            LangGraph 版 Agent（State / Node / Checkpointer / interrupt）
├── ★ main_lg.py             LangGraph 版 CLI 入口
├── ★ config.py              唯一配置与路径来源（12-factor：环境变量优先）
├── ★ bootstrap.py           服务装配（进程内只跑一次）+ MCP 启停
├──    static/index.html     前端（fetch + ReadableStream 解析 SSE）
│
├── ★ Dockerfile             镜像定义
├── ★ docker-compose.yml     服务编排（端口 / 挂载 / 重启策略）
├── ★ .dockerignore          构建时排除密钥 / 数据库 / 缓存
│
├── tools/
│   ├── rag_tools.py       ★ 混合检索：BM25 + 向量 + RRF + Cross-Encoder 精排
│   ├── bm25_tools.py      ★ 手写 BM25（jieba 分词 + TF/IDF + 长度归一化）
│   ├── chunk_utils.py     ★ 分块：固定切分 + 句子级切分（L2 递归分块思路）
│   ├── local_tools.py       文件读写删
│   └── memory_tools.py      长期记忆（JSON 持久化）
│
├── eval/                  ★ 评估闭环
│   ├── ragas_golden.py      10 条 golden 测试集（问题 + 标准答案 + 应命中子串）
│   ├── m1_recall_eval.py    检索指标：Recall@K / MRR
│   ├── m2_generate.py       检索 → 生成
│   └── m2_judge.py          生成指标：手写 LLM-as-Judge
│
├── knowledge/               知识库（5 篇主题短文 + 1 篇论文）
│   └── embeddings.json      向量索引
└── tests/                   单元测试（24 个，`python -m unittest discover tests`）
```

---

## 🚀 快速开始

```bash
# 0. 选对解释器（最容易踩的坑）
#    必须用装了 torch / sentence-transformers / jieba 的解释器。
#    本机可用的是 conda 环境 pytorch_env：
#        E:\deeplearning\envs\pytorch_env\python.exe
#    直接敲 `python` 有风险：PATH 上的 python 可能命中 Windows Store 的
#    0 字节占位存根，现象是「敲下去没有任何输出就直接退出」——
#    那不是本项目代码的问题。先 conda activate pytorch_env 再往下走。

pip install -r requirements.txt

# 1. 配置密钥（.env 已被 .gitignore 忽略，绝不进 git）
#    在【仓库根目录】创建 .env，格式见同目录的 .env.example：
#      DEEPSEEK_API_KEY=sk-你的key
#      DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
#      DEEPSEEK_MODEL=deepseek-v4-flash

# 2. 构建向量索引
python build_index.py

# 3. 冒烟测试（24 个单元测试，不需要 API Key）
python -m unittest discover tests

# 4. 启动方式 A —— CLI 交互（最简单）
python main.py          # 手写 ReAct 版
python main_lg.py       # LangGraph 版：会话持久化 + 流式打字机 + 工具人工审批

# 5. 启动方式 B —— HTTP 服务（推荐，带流式前端）★
#    必须在 assistant/ 目录下执行
uvicorn server:app --host 0.0.0.0 --port 8000
#    开发时加 --reload 改代码自动重启
#    → 浏览器打开 http://127.0.0.1:8000/      极简打字机前端
#    → curl http://127.0.0.1:8000/health      健康检查
#    → POST /chat        一问一答（非流式）
#    → POST /chat/stream SSE 流式（前端用的就是这个）
#    → POST /chat/resume 审批答复，从断点继续

# 6. 启动方式 C —— Docker（见下一节）
docker compose up -d

# 7. 跑评估
python eval/m1_recall_eval.py     # 检索指标
python eval/m2_judge.py           # 生成指标
```

> **三种启动方式的关系**：`main.py` / `main_lg.py` 是**本地调试**入口；
> `server.py` 是**对外服务**入口，它复用同一套 `bootstrap.build_agent()`，
> 区别只是把 `print` 换成了 SSE 事件流。

---

## 🐳 Docker 部署

### 一键启动

```bash
# 1. 准备 compose 变量（.env 在 docker-compose.yml 同级，已被 .gitignore 忽略）
#    HF_CACHE = 宿主机 HuggingFace 缓存目录  → 复用已下好的模型，容器内零重下
#    KB_DATA  = 宿主机运行时数据目录        → 记忆 / checkpoint / 笔记
#    ⚠️ 两个路径都必须落在【固定盘】上，不能是可移动盘（U 盘/读卡器）—— 见踩坑 8

# 2. 构建 + 启动
docker compose up -d

# 3. 验证
curl http://127.0.0.1:8000/health
#   {"status":"ok"}

# 4. 看实时日志（靠 PYTHONUNBUFFERED=1 才不延迟，见下表）
docker compose logs -f

# 5. 停止（数据留在 KB_DATA，不丢）
docker compose down
```

### 文件清单

| 文件 | 作用 |
|---|---|
| `Dockerfile` | 镜像定义：`python:3.10-slim` + 依赖 + 代码 |
| `docker-compose.yml` | 服务编排：端口 / 环境变量 / volume 挂载 / 重启策略 |
| `.dockerignore` | 构建时**不打包**的东西（密钥 / 数据库 / 缓存 / 测试 / 旧索引） |
| `.env`（与 compose 同级，gitignored） | 只放 **compose 变量替换**用的两个路径，不放密钥 |

### 实测结果

| 项 | 值 |
|---|---|
| 镜像 | `kb-assistant:latest`，**2.05 GB** |
| 构建耗时 | **约 6 分钟**（换国内源后；换源前 30 分钟以上没跑完） |
| 容器 | `kb`，`Up`，`restart: unless-stopped` |
| `GET /health` | `{"status":"ok"}` |
| `GET /` | 200，3103 bytes |
| 模型下载次数 | **0** —— `/models` 直接复用宿主机 HF 缓存 |
| 首启耗时 | 数十秒（含加载向量模型 + reranker + MCP 子进程） |

### Dockerfile 关键点

| 写法 | 为什么必须这样 |
|---|---|
| `FROM python:3.10-slim` | 与开发机 3.10.18 对齐；slim 比完整版小约 700 MB |
| `ENV PYTHONUNBUFFERED=1` | Python 默认缓冲 stdout → **不设它 `docker logs` 看不到实时日志**（启动日志、报错全被攒着） |
| `ENV HF_HOME=/models` | 把模型缓存指向挂载点，容器重建也不重下模型 |
| `COPY requirements.txt .` **在** `COPY . .` **之前** | 分层缓存：只改代码时这一层命中缓存，不用重装依赖 |
| 三个**独立** `RUN pip install` | 不 `&&` 串成一条 —— 某步失败时，前面成功的层仍留在缓存，重试不必重下 |
| `torch` 单独走 `--index-url .../whl/cpu` | PyPI 默认拉 CUDA 版（2 GB+），本项目 0 显存，装 CPU 版 |
| `CMD uvicorn --host 0.0.0.0` | 默认 `127.0.0.1` 是**容器内部**回环，宿主机/外网连不上；容器内服务一律 `0.0.0.0` |
| `ARG PIP_INDEX` / `ARG TORCH_INDEX` | 源参数化，可用 `--build-arg` 临时切阿里云等镜像 |

### 挂载（volume）表

| 宿主机 | 容器内 | 用途 |
|---|---|---|
| `$HF_CACHE` | `/models` | HuggingFace 模型缓存（**复用**，容器内零重下） |
| `$KB_DATA/memory.json` | `/app/memory.json` | 长期记忆 |
| `$KB_DATA/lg_checkpoints.db` | `/app/lg_checkpoints.db` | LangGraph 会话 checkpoint（thread 持久化） |
| `$KB_DATA/my_notes` | `/app/my_notes` | 文件工具的工作目录 |

> ⚠️ **单文件挂载前，宿主机上的文件必须先存在**，否则 Docker 会替你把它建成一个**目录** —— 见踩坑 7。

### 镜像加速（国内必做）

| 问题 | 现象 | 解法 |
|---|---|---|
| Docker Hub 连不上 | `dial tcp ...:443: connectex: 连接尝试失败` | `daemon.json` 加 `registry-mirrors` |
| PyPI 直连极慢 | numpy 16.3 MB 下了 6 分钟（≈45 KB/s） | 换清华源 → **3~5 MB/s，约 90×** |
| HuggingFace 连不上 | 模型下不来 | `HF_ENDPOINT=https://hf-mirror.com`（已写进 Dockerfile） |
| Docker 数据占 C 盘 | `docker_data.vhdx` 涨到 6 GB | Docker Desktop → 设置里改 **Disk image location** 到数据盘 |

---

## 🔬 评估方法

**为什么自己写评估脚本，而不用 RAGAS 库？**
RAGAS 0.4.x 硬依赖 `langchain-google-vertexai`，在离线/受限环境装不上。于是按同样的方法论**手写了 LLM-as-Judge**（概念与 RAGAS 一致，但更可控、可调试）。

**M1 检索指标**

| 指标 | 定义 |
|---|---|
| Recall@K | 前 K 个召回结果里出现"应命中 chunk"的题目占比 |
| MRR | 平均倒数排名 `1/rank`（第 1 名=1.0，第 2 名=0.5，未命中=0） |

> Recall 管"能不能找到"，MRR 管"排得够不够靠前"。二者互补。

**M2 生成指标**：用一个严格裁判 prompt 让 LLM 输出单行 JSON（`faithfulness` / `answer_relevancy` / `correctness`），打分范围 0~1。

**Golden 测试集设计**：10 条问题，5 条中文主题题（中文问中文）+ 5 条论文题（**中文问英文论文**，专门测跨语种语义检索）。判命中用"独特子串匹配"而非 chunk 下标，因此**对分块策略无感** —— 换分块方案后评估仍然有效。

---

## 🕳 工程踩坑记录（最有价值的部分）

### 1. 分块质量 >> 检索算法

最初的切分是 `text[start:start+300]` 硬切，结果**单词被拦腰切断、期刊元数据/作者邮箱/参考文献混进知识库、PDF 提取噪声没清**。最有代表性的一块：

```
'March 2011 / Available online 7 April 2011 / Keywords: Finite element
 Assumed stress hybrid methods are known to improve the performance ...'
```

正确答案埋在这种"元数据垃圾堆"中间，任何检索器都抓不住。

**改成"句子级切分 + 累积合并 + 超长降级"后，什么都没动检索算法，Recall@3 就从 0.80 变 1.00。**

> **教训：RAG 效果不好时，先怀疑切片，别急着换模型。**

### 2. 递归分块：能用大边界就别用小边界

分隔符优先级：`段落 → 换行 → 句子 → 词 → 字符`。只有当"上一级切完还超长"时才降级。

实测发现论文里存在一个 **874 字符的"句子"**（标题+作者+单位块，通篇没有句号），必须降级到换行才切得开（切完最长行 105 字符）。

> **注意**：把 `\n` 直接加进切分模式是错的 —— 那样每个硬换行都成边界，块会停在句子中间。

### 3. BM25 的边界：它只认"字面"

BM25 是**词面匹配**。中文问题搜英文文档时，中英文 token 零交集 → **关键词路贡献为 0**，全靠向量路（多语言模型把中英映射到同一空间）。

实测：中文 query 时 BM25 甚至会把**中文短文排到第 1**（因为匹配了"是/的"这种高频词）—— 属于假阳性，靠 RRF + reranker 兜底。

> **跨语种检索要么靠多语言向量模型，要么做查询改写（中文问题 → 英文关键词）让 BM25 也能用。**

### 4. 跨进程通信必须显式指定编码

MCP 客户端用 `subprocess.Popen(..., text=True)` —— `text=True` 会用**系统默认编码**（中文 Windows 上是 GBK）解码子进程输出。一旦 `PYTHONIOENCODING` 被改（IDE、Docker、某个库都可能改），两端编码不一致就会崩，报错还极具误导性：

```
'gbk' codec can't decode byte 0xa1 in position 92: illegal multibyte sequence
```

**修复**：两端都显式钉死 UTF-8（客户端 `Popen(encoding="utf-8")`，服务端 `sys.stdin/stdout.reconfigure(encoding="utf-8")`）。

> **教训：跨进程/跨网络的 I/O 永远显式指定编码，不要依赖环境默认值。**

### 5. 导入风格必须全项目统一

项目里一度混用两种风格：包导入（`from tools.rag_tools import ...`）和 sys.path hack + 顶层导入（`import rag_tools`）。结果是**同一个 `rag_tools.py`，被 `main.py` 导入时正常，被 `build_index.py` 导入时崩**（相对导入 `.bm25_tools` 在顶层模块语境下无父包）。

**统一为项目内部的包导入**，外部脚本用**文件相对定位**把项目根加入 sys.path（不写死绝对路径，保证可移植）。

### 6. 密钥与代码分离

一开始 **17 个文件**都硬编码了同一个 API Key。抽成：

```
.env                ← 真实密钥（被 .gitignore 忽略）
.env.example        ← 模板（进 git，供人参考格式）
common_config.py    ← 唯一的读取入口
```

换 key 只需改 `.env` 一行。

> **⚠️ 注意**：早期提交历史里仍留有旧 key。**推送到 GitHub 之前必须先在平台后台作废旧 key、生成新 key。**

### 7. bind-mount：源文件不存在时，Docker 会**默默建一个目录**

`-v 宿主机路径:容器路径` 的规则是：**源路径不存在时它不报错，而是替你创建一个同名目录**（它默认你想挂目录）。

于是灾难链是：

| 步骤 | 发生了什么 |
|---|---|
| 1 | compose 写 `E:\kb-data\memory.json:/app/memory.json` |
| 2 | 但 `E:\kb-data\memory.json` **还不存在** |
| 3 | Docker 在宿主机建了个**目录** `E:\kb-data\memory.json\` |
| 4 | 挂进容器 → `/app/memory.json` 是个**目录** |
| 5 | 代码 `open("/app/memory.json")` → `IsADirectoryError [Errno 21]` |

**实测踩过**：`lg_checkpoints.db` 当时被 Docker 变成了一个空目录。

> **最阴的地方**：**现象出现在容器里，根因在宿主机。** 看容器日志会以为是代码 bug，其实宿主机上多了个莫名其妙的目录。

**两种修法：**

| 方法 | 做法 | 适用 |
|---|---|---|
| 先建文件 | `New-Item -ItemType File -Path E:\kb-data\memory.json` | 挂单个已知文件（本项目） |
| 改挂目录 | 挂 `E:\kb-data:/app/data`，文件在容器内自然生成 | 数据文件多 / 名字会变（**更推荐**） |

> **面试讲法**：声明式配置工具（Docker / K8s）遇到"资源缺失"，行为可能是**静默创建而不是报错**。所以部署前要么**显式初始化挂载源**，要么用 entrypoint 做 preflight 检查。K8s 里 `subPath` 挂单文件踩的是**同一个坑**。

### 8. `H:` 是「可移动盘 + FAT32」——两个独立问题撞一起

排查 `docker compose up` 时报：

```
error mounting ".../memory.json" ... not a directory:
Are you trying to mount a directory onto a file (or vice-versa)?
```

**根因是两个问题叠加，得分开讲。**

**问题 A：可移动盘 → 容器里根本看不见**

Docker Desktop 在 Windows 上跑在 WSL2 的轻量 Linux 虚拟机里。宿主机磁盘要进容器，得**先被共享进那台虚拟机**——而 Docker Desktop **只自动共享固定盘（Fixed）**，可移动盘（U 盘 / 读卡器 / 移动硬盘）默认**不共享**。所以 `H:\...` 这个路径，**在容器看来压根不存在**。

> 注意这跟权限无关，是"**看不见**"。Linux 里没有 `C:/D:/E:` 的概念，共享盘以 `/mnt/host/c/...` 之类路径出现。

**问题 B：FAT32 → 单文件最大 4 GiB − 1 字节**（本机实测）

| 在 `H:` 上创建 | 结果 |
|---|---|
| **4095 MB** 文件 | ✅ 成功 |
| **4096 MB** 文件 | ❌ `There is not enough space on the disk` |

而 `docker_data.vhdx` 现在就有 **6137 MB** —— **超过 4 GB**。所以就算 Docker 肯共享 H 盘，那虚拟磁盘文件也**根本放不下**。

FAT32 的附带问题（都不报错，只是行为诡异）：

| FAT32 缺什么 | 后果 |
|---|---|
| 单文件 ≤ 4 GiB − 1 | vhdx、大模型权重存不下 |
| 无 Unix 权限位 | 挂进容器后脚本可能不可执行（`chmod` 无效） |
| 无符号链接 | `node_modules` / 部分 pip 包会崩 |
| 无日志（journaling） | 断电或直接拔出易整盘损坏 |
| 时间戳精度 2 秒 | 增量构建、文件监听（HMR）判断失准 |
| 读写慢 | 构建、建索引都慢 |

> **本项目结论**：**代码放哪都行，但"容器要读的路径"必须在固定盘。** 于是把运行时数据搬到 `E:\kb-data`，代码留在 `H:`。

> **面试讲法**：**「容器看到的文件系统」≠「你在 Windows 里看到的盘」**，中间隔着 WSL2 共享层，而共享层对盘的类型有要求。排查顺序：**先确认源路径在容器里存不存在**（`docker run --rm -v <路径>:<容器路径> alpine ls -la <容器路径>`），确认"看得见"之后，再谈权限和路径对不对。

---

## ⚠️ 已知限制

- **知识库偏小**：6 篇文档 / 31 个 chunk，Recall@3 已到 1.00（**饱和**），继续提升需要扩库 + 加更难的问题。
- **中文问英文**：关键词路零贡献（见踩坑 3）。
- **块 0/1 仍是元数据碎片**：论文标题/作者/单位块，PDF 提取还把它们粘成了无空格的一坨，语义偏弱。
- **检索是 O(N) 暴力遍历**：未接向量库（FAISS / Qdrant / pgvector），语料一大会慢。
- **单进程 / 无鉴权**：接口没有认证，`SqliteSaver` 在多 worker 并发写下会出现 `database is locked`；无限流。
- **Langfuse cost 显示 $0**：`deepseek-v4-flash` 不在 Langfuse 内置价格表里（**token 数正常**，够用）。
- **评估集判分用"子串匹配"**：chunk 边界变化不影响，但如果同一句话在多个 chunk 出现，会误判。

## 🗺 后续计划

- [ ] 扩知识库 + 更刁钻的 golden 问题（提高评估区分度）
- [ ] **语义分块**（用 embedding 计算相邻句子相似度，在语义断层处切）
- [ ] **查询改写**：中文问题 → 英文关键词，让 BM25 具备跨语种能力
- [x] **Langfuse 可观测**：延迟 / token / 工具调用链
- [x] **FastAPI + SSE 流式接口 + Docker 部署**
- [ ] **多用户隔离 / 限流 / 鉴权**（当前是单进程单用户假设）
- [ ] 向量库迁移（Qdrant / pgvector），支持 HNSW 索引 + 元数据过滤
- [ ] 缓存层（相同/相似 query 直接命中，省 API 钱和延迟）
- [ ] 云部署 + 对外 demo

---

## 🧰 技术栈

| 层 | 选型 |
|---|---|
| LLM | DeepSeek API（`deepseek-v4-flash`） |
| 向量模型 | `paraphrase-multilingual-MiniLM-L12-v2`（多语言，中英互通） |
| 重排模型 | `BAAI/bge-reranker-v2-m3`（Cross-Encoder，中英） |
| 关键词检索 | **手写 BM25**（jieba 分词 + TF-IDF + 长度归一化） |
| 融合策略 | RRF（Reciprocal Rank Fusion） |
| Agent | **手写 ReAct 循环**（`agent_core.py`，不依赖框架） |
| Agent 编排 | **LangGraph**（`agent_lg.py`：State / Reducer / 条件边 / Checkpointer / `interrupt()`） |
| 工具协议 | MCP（Model Context Protocol，JSON-RPC over stdio） |
| 评估 | **手写 LLM-as-Judge** + Recall@K / MRR |
| 服务 | FastAPI + Uvicorn（SSE 流式 `/chat/stream`） |
| 可观测 | Langfuse（`langfuse.langchain.CallbackHandler`） |
| 持久化 | JSON（记忆 + 向量索引）+ SQLite（LangGraph checkpoint） |
| 部署 | Docker + docker-compose（镜像 `kb-assistant`） |