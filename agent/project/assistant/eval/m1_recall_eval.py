import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys, re
# 让 rag_tools 可导入（assistant 目录）+ 让 ragas_golden 可导入（当前脚本所在目录会自动在 path）
# 用【文件相对定位】把项目根(assistant)加入 sys.path —— 不写死绝对路径，项目可移植
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.rag_tools import init_rag, hybrid_search             # 包导入
from ragas_golden import golden


KNOWLEDGE_DIR = r"H:/ai-learning/agent/project/assistant/knowledge"

def norm(s):
    """规整空白：把任意连续的空白(含换行)压成单个空格。⚠️ 论文 chunk 里有 \n，必须做这步。"""
    return re.sub(r"\s+", " ", s).strip()

def first_hit_rank(parts, gold_norm):
    """在检索回来的 parts 里，找 gold 第一次出现的排名（1-based）；找不到返回 None。

    parts: 每个被召回 chunk 的文本列表（顺序 = 检索顺序，第 1 个最重要）。
    """
    # ============ 你写这里 ============
    # 遍历 parts，逐个 norm(part)，看 gold_norm 在不在里面；
    # 在的话：返回"它是第几个"(从 1 开始)；都不在：返回 None
    # =================================
    for i in range(len(parts)):
        if gold_norm in norm(parts[i]):
            return i+1
    return None


def main():
    init_rag(KNOWLEDGE_DIR)          # 加载 MiniLM + bge-reranker（要点时间）
    k = 3                             # 取前 k 个召回
    total = hits = 0
    rr_sum = 0.0
    for item in golden:
        gold_norm = norm(item["gold"])
        text = hybrid_search(item["question"], top_n=k)   # 返回拼好的字符串
        parts = text.split("\n---\n")                     # 拆成各个被召回 chunk
        rank = first_hit_rank(parts, gold_norm)
        total += 1
        if rank:
            hits += 1
            rr_sum += 1.0 / rank
            print(f"[命中 rank={rank}] {item['question'][:26]}")
        else:
            print(f"[未命中] {item['question'][:26]}")
    print(f"\nRecall@{k} = {hits}/{total} = {hits/total:.2f}")
    print(f"MRR        = {rr_sum/total:.3f}")

if __name__ == "__main__":
    main()