import sys, os
sys.path.insert(0, r"H:\ai-learning\agent\project\assistant")
from tools import rag_tools

KNOWLEDGE_DIR = r"H:\ai-learning\agent\project\assistant\knowledge"
rag_tools.init_rag(KNOWLEDGE_DIR)

query = "LSTM"
query_vec = rag_tools._model.encode(query)

# 模仿你的小实验：算出每个 chunk 的相似度，列出来排序
scores = []
for i, doc_vec in enumerate(rag_tools._doc_embeddings):
    sc = rag_tools.cosine_similarity(query_vec, doc_vec)
    chunk = rag_tools._emb_data["chunks"][i]
    scores.append((i, chunk["source"], sc))

scores.sort(key=lambda x: x[2], reverse=True)

k = 5
print(f"共有 {len(scores)} 个 chunk，取前 {k}：")
for rank, (i, src, sc) in enumerate(scores[:k], 1):
    print(f"  [{rank}] idx={i}  source={src}  score={sc:.4f}")

print("=== 第 0 个 chunk 的字典 ===")
print(rag_tools._emb_data["chunks"][0])

print("=== 第 0 个 chunk 的向量 ===")
print(rag_tools._doc_embeddings[0])

print("=== _emb_data 有哪些 key ===")
print(rag_tools._emb_data.keys())    

print("第0个chunk的source:", rag_tools._emb_data["chunks"][0]["source"])
print("第0个chunk的向量长度:", len(rag_tools._doc_embeddings[0]))

print("模型类型:",type(rag_tools._model))
print("向量形状:",rag_tools._doc_embeddings.shape)

print("混合检索:\n", rag_tools.hybrid_search("注意力机制"))
print("==========")
print("纯向量:\n", rag_tools.semantic_search("注意力机制"))