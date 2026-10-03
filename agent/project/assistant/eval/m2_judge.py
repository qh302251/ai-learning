# -*- coding: utf-8 -*-
"""M2b — LLM-as-Judge：给"回答"打 忠实度 / 回答相关性 分（0~1）"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys, json, re
import requests
# 用【文件相对定位】加入 sys.path —— 不写死绝对路径，项目可移植
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))                    # → assistant/  (tools 包)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")))  # → 仓库根      (common_config)
from tools.rag_tools import init_rag, hybrid_search
from common_config import API_KEY
from ragas_golden import golden

DEEPSEEK = {
    "api_key": API_KEY,
    "api_url": "https://api.deepseek.com/v1/chat/completions",
    "model": "deepseek-v4-flash",
}
KNOWLEDGE_DIR = r"H:/ai-learning/agent/project/assistant/knowledge"
TOP_N = 3

def call_llm(messages, max_tokens=2048):
    r = requests.post(
        DEEPSEEK["api_url"],
        headers={"Authorization": f"Bearer {DEEPSEEK['api_key']}", "Content-Type": "application/json"},
        json={"model": DEEPSEEK["model"], "messages": messages, "max_tokens": max_tokens},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def generate_answer(question, context):
    msgs = [
        {"role": "system", "content": "你是知识库问答助手。只能根据下面的知识库内容回答；知识库里没有的，明确说'知识库未提及'，不要编造。"},
        {"role": "user", "content": f"知识库内容：\n{context}\n\n问题：{question}\n\n请用中文简洁回答。"},
    ]
    return call_llm(msgs)

def judge(question, context, answer):
    msgs = [
        {
        "role": "system", 
        "content": """
        你是一个严格的评判裁判。你的回答结果只能输出一行JSON：{"faithfulness": 0.x, "answer_relevancy": 0.y},
        faithfulness: 回答是否**只基于 context**、没编造/没掺入 context 没有的信息,
        answer_relevancy: 回答是否**针对问题**、没跑题、没答非所问,
        不要因为回答通顺、像人话、或大致相关就给高分。
        只有真正"每句都有依据/完全切题"才给 1.0。
        """
        },
        {"role": "user", "content": f"知识库内容：\n{context}\n回答：\n{answer}\n\n问题：{question}\n"},
    ]
    raw = call_llm(msgs)
    
    try:
        m = re.search(r'\{.*\}', raw, re.S)   # ① 先抓出 {...} 那一段（模型有时会包 ```json 或加解释）
        d = json.loads(m.group(0))
        f = float(d["faithfulness"])          # ③ 取两个分
        r = float(d["answer_relevancy"])
        return f,r
    except Exception:
        print("⚠️ JSON 解析失败，raw =", raw[:200])   # 让你看清楚模型输出了啥
        return 0.5, 0.5

def judge_correctness(question, ground_truth, answer):
    msgs = [
        {"role": "system", "content": """你是一个严格的评估裁判。
只输出一行 JSON：{"correctness": <0~1>}
correctness = 1.0：回答包含了标准答案的**所有关键事实**。
correctness = 0.6：回答**遗漏了部分关键点**，但没答错。
correctness = 0.0：回答**答错**或与标准答案**矛盾**。
不要因为回答通顺就放宽；遗漏了关键点就要扣分。"""},
        {"role": "user", "content": f"问题：{question}\n标准答案：{ground_truth}\n回答：{answer}\n请按规则打分，只输出 JSON。"},
    ]
    raw = call_llm(msgs)
    try:
        m = re.search(r'\{.*\}', raw, re.S)
        d=json.loads(m.group(0))
        c = float(d["correctness"])
        return c
    except Exception:
        return 0.5
    # ↓ 你写：从 raw 抽 JSON → json.loads → 取 correctness → try/except 兜底
    #    （跟你在 judge() 里写的解析手法一模一样，直接复用）
    ...



def main():
    init_rag(KNOWLEDGE_DIR)
    fs, rs, cs = [], [], []
    for item in golden:
        q = item["question"]
        ctx = hybrid_search(q, top_n=TOP_N)
        ans = generate_answer(q, ctx)
        f, r = judge(q, ctx, ans)
        c = judge_correctness(q, item["ground_truth"], ans)
        fs.append(f); rs.append(r); cs.append(c)
        print(f"[{q[:20]}] faith={f:.2f} rel={r:.2f} corr={c:.2f}")
    print(f"\n平均 忠实度 = {sum(fs)/len(fs):.3f}")
    print(f"平均 回答相关性 = {sum(rs)/len(rs):.3f}")
    print(f"平均 correctness = {sum(cs)/len(cs):.3f}")

if __name__ == "__main__":
    main()