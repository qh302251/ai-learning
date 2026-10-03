# 大模型开发学习计划

> 硬件：RTX 2060 (6GB VRAM) | CUDA 13.2
> 前置：已完成 Agent 开发全流程，熟悉 Python、API 调用、Embedding

---

## 第一阶段：Prompt Engineering 系统化（预计 4-6 小时）

打好使用大模型的基础功，理解模型能力的边界。

- [x] **基础 Prompt 技巧**
  - [x] Zero-shot / Few-shot / Chain-of-Thought 原理与适用场景
  - [x] Role Prompting 与格式控制（JSON、Markdown 结构化输出）
- [x] **进阶 Prompt 模式**
  - [x] Tree-of-Thought、ReAct（已掌握，复习即可）
  - [x] 自一致性（Self-Consistency）、思维链触发词
- [x] **实用技巧**
  - [x] 长上下文处理（摘要、分段、滑动窗口）
  - [x] 防止 Prompt Injection / Jailbreak 的基本措施
- [x] **练习**：用 DeepSeek API 写 5 个不同场景的 Prompt，对比效果差异

## 第二阶段：环境搭建与基础工具链（预计 2-3 小时）

- [x] 安装核心依赖：PyTorch + Transformers + Datasets + Accelerate + bitsandbytes
- [x] 确认 CUDA 可用，测试 6GB 下能跑多大的模型
  - float32: 1B=3.7GB → 7B 跑不动
  - float16: 3B=5.6GB → 7B 勉强
  - int8/4-bit: 7B~5GB → ✅ 唯一可行方案
- [x] 熟悉 HuggingFace Hub：模型下载、缓存管理、镜像源配置（已会）
- [x] 了解 Model Card 阅读方法：参数量、许可、硬件要求、评测指标

## 第三阶段：模型推理与量化（预计 4-6 小时）

先学会跑模型，再谈微调。

- [x] **Transformers 基础推理**
  - [x] AutoModel / AutoTokenizer 加载与使用
  - [x] generate 参数详解（temperature, top_p, repetition_penalty, do_sample）
  - [x] Padding / Attention Mask / max_length / truncation 等基础概念
  - [x] BPE 分词原理：按字频合并 token，只看频率不懂语义
  - [x] Special Tokens（BOS / EOS / PAD）作用
- [x] **Transformer 架构原理（基础理论）**
  - [x] Self-Attention & QKV：Q 问谁跟我有关，K 回答我是谁，V 提取内容
  - [x] Multi-Head Attention：多组 QKV 同时找多种关系
  - [x] FFN & LayerNorm：Attention 是检索，FFN 是思考
  - [x] RoPE 旋转位置编码：用旋转角度表示词顺序
  - [x] Prefill vs Decode & KV Cache：一次性预填充 + 逐词生成 + 缓存加速
- [x] **量化基础**
  - [x] 为什么需要量化：6GB 跑 7B 模型必须 4-bit
  - [x] bitsandbytes 4-bit / 8-bit 量化加载（已装 bitsandbytes、配置了 BnBConfig）
  - [x] 量化对推理质量和速度的影响
  - [x] **4-bit 量化原理深入**：float → 16级映射 → 反量化还原，精度损失来源，bfloat16 vs float32
- [ ] **推理加速（放到第七阶段部署再学）**
  - Flash Attention 原理与启用（RTX 2060 不兼容，硬件限制）
  - vLLM 部署入门（6GB 显存不足以流畅运行）
  - Ollama 本地部署体验（对开发学习者非必要，部署阶段再学）
- [x] **练习**：用 Transformers 加载 Qwen2.5-7B（4-bit，已保存到 E 盘秒加载），交互式推理脚本（chat.py）

## 第四阶段：数据集构建（预计 5-8 小时）

数据质量 > 模型大小，这是最值得花时间的环节。

- [x] **数据集格式**
  - [x] 对话格式（ShareGPT / Alpaca / ChatML）
  - [x] 指令微调数据构造（Instruction → Input → Output）
  - [x] 数据集拆分：train / validation / test
- [x] **HuggingFace Datasets 库**
  - [x] 加载开源数据集（shibing624/alpaca-zh，48818 条）
  - [x] 预处理、映射（map）批量处理
  - [x] Tokenize 函数编写（Alpaca → ChatML → token IDs）
- [x] **Datasets 库补充操作**
  - [x] 加载本地数据集（从 json 文件）
  - [x] filter 过滤筛选数据（回答长度 > 50 字）
  - [x] save_to_disk / load_from_disk 保存与加载，判断缓存是否存在
- [x] **长度与分布分析**
  - [x] 最短 35 / 最长 867 / 平均 171 token
- [x] **数据清洗与增强**
  - [x] 去重（hash + Dataset.filter 去重，0 条重复）
  - [x] 去噪声（筛掉指令 < 5 字 + 模板残留，共筛掉 637 条）
  - [x] 长度过滤（已有回答长度 > 50 字过滤）
- [x] **练习**：用 map() 将 Alpaca-zh 转为 ChatML 格式并 tokenize

## 第五阶段：LoRA / QLoRA 微调（预计 8-12 小时）

核心技能，重点是理解原理而不是跑通脚本。

- [x] **PEFT / LoRA 原理**
  - [x] Full Fine-tuning vs LoRA vs QLoRA 的区别
  - [x] rank（r）、alpha、target_modules 的含义与选择
  - [x] 为什么 LoRA 能 work：本质是低秩近似
- [x] **微调流程**
  - [x] 加载 base model + tokenizer（float16 1.5B）
  - [x] 配置 LoRA（选择 target modules）
  - [x] Training Arguments / SFTConfig 配置
  - [x] SFTTrainer 训练与监控 loss（1.5B 训练完成：3h28m，loss 1.71→1.60）
  - [x] 合并 / 保存 LoRA 权重（test_LoRA.py 挂载推理 + compare_models.py 对比效果）
- [x] **常见问题与调试**
  - [x] OOM（Out of Memory）的解决策略
  - [x] Loss 不下降 / 过拟合 / 灾难性遗忘
  - [x] 验证集评估与 checkpoint 选择
- [x] **练习**：用 QLoRA 微调 Qwen2.5-7B 或 LLaMA-3-8B，做一个个性化助手（已决定跳过 7B，用 1.5B 完成完整流程教学：训练→保存→推理测试→checkpoint对比）

## 第六阶段：模型评估（预计 3-5 小时）

没有评估的微调就是盲调。

- [x] **自动评估**
  - [x] Perplexity（困惑度）计算（ppl_eval.py，发现数据泄露问题并公平化）
  - [x] 用更强模型打分（LLM-as-Judge，judge.py 用 DeepSeek API 裁判：原始37 vs 微调36）
  - [x] Benchmark 评估（MMLU, CEVAL, GSM8K 等概念，跳过实跑）
- [x] **人工评估**
  - [x] 评估维度：有用性、安全性、格式遵循
  - [x] A/B 测试对比微调前后效果（compare_models.py 多问题对比）
- [x] **练习**：对第五阶段微调后的模型做评估（PPL + LLM-as-Judge + 人工对比三种方法综合评估）

## 第七阶段：模型部署（预计 4-6 小时）

把模型变成可用的服务。

- [x] **vLLM 部署（概念，硬件不支持实装）**
  - [x] 概念：OpenAI 兼容 API、吞吐量测试（RTX 2060 6GB 跑不动 vLLM，只讲概念）
  - [x] LoRA 权重热加载 / 吞吐量测试（未实装，硬件限制）
- [x] **Ollama 部署**
  - [x] GGUF 概念（safetensors vs GGUF、量化）
  - [x] 安装 Ollama、Modelfile 编写、ollama create/list/run/show/rm 全流程
  - [x] 模型目录迁移（OLLAMA_MODELS 环境变量，解决 C 盘占用）
  - [x] 尝试 GGUF 转换微调模型：遇到 Ollama 0.32.5 bug（转换丢失特殊 token，模型刷 @）→ 改用 Python 部署
- [x] **练习**：将微调后的模型部署成服务并用 curl 测试 API（改用 Python 方案）
  - [x] Flask 将微调模型包装成 OpenAI 兼容 API（/v1/chat/completions）
  - [x] curl 测试 API（UTF-8 文件绕过 PowerShell 中文编码坑）
  - [x] 网页聊天界面（浏览器直接聊天）

## 可选进阶方向（根据需要选学）

- **DPO 偏好对齐**：比 RLHF 更简单的对齐方法，需要偏好数据集
- **RAG 深化**：你已经有基础，可以深入 Chunk 策略、重排序、Graph RAG
- **Multi-Modal**：LLaVA 等视觉语言模型的微调
- **模型量化进阶**：GPTQ / AWQ / GGUF 的区别与选择

---

## 路线图总览

```
第1周               第2周               第3周               第4周
Prompt 工程  ──→  推理与量化  ──→  数据集构建  ──→  QLoRA 微调
                                          ↘
                                           评估与部署  ←──  调优迭代
```

## 推荐的开源模型（6GB VRAM 友好）

| 模型 | 参数量 | 4-bit 显存 | 说明 |
|------|--------|-----------|------|
| Qwen2.5-7B | 7B | ~5GB | 中文最强，推荐首选 |
| LLaMA-3-8B | 8B | ~5.5GB | 英文强，中文一般 |
| DeepSeek-V2-Lite | 16B | ~6GB (临界) | 可能爆显存 |
| ChatGLM3-6B | 6B | ~4.5GB | 轻松跑，效果中上 |

**推荐首选：Qwen2.5-7B**，中文生态好、社区活跃、文档齐全。

## 项目建议（二选一，作为简历亮点）

1. **领域微调助手**：选择一个你熟悉的领域（如数学、法律、代码），搜集数据微调出一个专用助手
2. **中文风格迁移**：用 LoRA 微调模型模仿特定写作风格（如古龙/鲁迅），体现对 LoRA 原理的理解
