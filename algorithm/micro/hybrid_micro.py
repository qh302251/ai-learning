import math, re
from collections import Counter
import numpy as np

def tokenize(text):
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())

# 4 个 chunk，既有文本（给关键词路用），也有向量（给语义路用）
chunks = {
    "cnn":   {"text": "卷积神经网络 处理 图像 卷积 池化",                 "vec": np.array([1.0, 0.0])},
    "transformer": {"text": "Transformer 自注意力 机制 NLP 改变",          "vec": np.array([0.9, 0.1])},
    "rnn":   {"text": "循环神经网络 RNN 序列 文本 LSTM",                  "vec": np.array([0.1, 0.9])},
    "rl":    {"text": "强化学习 智能体 环境 奖励 Q-Learning",              "vec": np.array([0.0, 1.0])},
}
query = "注意力机制"
q_vec = np.array([1.0, 0.0])

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ① 关键词路：给每个 chunk 一个"命中词数"分 → 排序出 keyword_rank（按分从高到低的 chunk 名列表）
def keyword_rank(chunks, query):
    q = set(tokenize(query))
    scored = []
    for name, info in chunks.items():
        hit = sum(1 for w in set(tokenize(info["text"])) if w in q)
        scored.append((name, hit))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored]
# ② 向量路：用 cosine 对每个 chunk 打分 → 排序出 vector_rank
def vector_rank(chunks,query):
    pairs=[]
    for name,info in chunks.items():
        score=cosine(q_vec,info["vec"])
        pairs.append((name,score))
    pairs.sort(key=lambda x : x[1],reverse=True)
    return [n for n,_ in pairs]
# ③ RRF：把 keyword_rank 和 vector_rank 两路"名次"融合 → 按融合分从高到低取前 2
def rrf_fuse(ranked_lists, k=60):
    score = {}
    for ranks in ranked_lists:
        for rank, name in enumerate(ranks):
            score[name] = score.get(name, 0) + 1/(rank + k)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)

keyword_rank=keyword_rank(chunks,query)
vector_rank=vector_rank(chunks,query)
rrf_score=rrf_fuse([keyword_rank,vector_rank])
print(rrf_score[:2])