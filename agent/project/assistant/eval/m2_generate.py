# -*- coding: utf-8 -*-
"""M2a — 检索 + 生成回答（还没打分）。先看 DeepSeek 到底生成了什么。"""
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

def call_llm(messages, max_tokens=4096):
    """非流式调 DeepSeek，返回 content 字符串（自动忽略 thinking）。"""
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

def main():
    init_rag(KNOWLEDGE_DIR)
    for item in golden:
        q = item["question"]
        ctx = hybrid_search(q, top_n=3)
        ans = generate_answer(q, ctx)
        print("=" * 60)
        print("Q:", q)
        print("A:", ans.strip()[:180])
        print("   标准答案:", item["ground_truth"][:60])

if __name__ == "__main__":
    main()