# 学习者偏好（必须遵守）

用户：川大计算数学硕士，PINN / 科学计算背景，做项目驱动式 AI 学习，目标是求职竞争力
（RAG / Agent / 大模型应用开发 / 数据 / 评估）。

硬件：RTX 2060 6GB，无云 GPU → **不走训练/推理优化**，走 API + 框架 + 工程。

---

## 一、进度更新格式（硬性要求）

**每次汇报进度，必须按这个顺序先说清 4 件事，一段都不能少：**

1. **更新的点在哪** —— 具体到文件、函数、行
2. **相较之前改进了什么** —— 用对比（表格最好），让人一眼看到差别
3. **为什么做这个改进** —— 解决什么真实问题；面试里怎么讲
4. **还差什么** —— 这一步没覆盖的、后面要补的

**然后才是下一步的内容。**

- 不要一上来就给代码、给新概念。
- 不要连续给多个新概念。

---

## 二、每次给多少（硬性要求）

- **一次只推进一小步**，给的东西要能在几分钟内消化完
- 用户原话：「你每次给我的内容不要太多了，我看着很累」「每次不要给我太多东西，一步一步给」
- 每一步要**讲细**，不要跳
- 表格是用户最能接受的表达形式
- 不要替他跑测试、不要替他改他的练习文件

---

## 三、教学契约：我教，我做

- 老师给：**概念 + 骨架 + 引导问题**
- **绝不直接给完整答案代码** —— 用户要自己写
- 用户原话：「别光给代码，没思路我怎么写」「别跳太大」
- 每写完一步，**等用户验证通过**再进下一步

---

## 四、环境事实（省得重复踩坑）

- Python：`E:\deeplearning\envs\pytorch_env\python.exe`（3.10.18）
- 跑脚本：`& E:\deeplearning\envs\pytorch_env\python.exe <file>`
- PowerShell 里设 `$env:PYTHONIOENCODING="utf-8"`
- 过滤噪音：`| Select-String -NotMatch 'pkg_resources|UserWarning|import pkg'`
- 读文件用 `[System.IO.File]::ReadAllText(...)`（`Get-Content -Raw` 会按 GBK 解错中文）
- 写文件用 `[System.IO.File]::WriteAllText($p, $c, [System.Text.UTF8Encoding]::new($false))`
- 临时脚本写 `.tmp\` 或 `$env:TEMP`，用完删掉
- 已装：langgraph 1.2.11 / langgraph-checkpoint 4.2.0 / langgraph-checkpoint-sqlite 3.1.1 /
  langchain 1.4.0 / langchain-openai 1.6.0 / ragas 0.4.3（**未装 pytest**，用 `python -m unittest`）

---

## 五、进度记在哪

- 路线与进度：`notes\最终版学习与求职路线.md`
- 面试话术：`notes\面试提词卡.md`
- 错题：`notes\错题库.md`
- Agent 练习：`agent\micro\`
- 项目：`agent\project\assistant\`

**每次推进完，回来更新 `notes\最终版学习与求职路线.md` 的进度。**