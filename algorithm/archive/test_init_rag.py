import sys
# 让脚本能找到 rag_tools.py（它在 assistant/tools/ 下）
sys.path.insert(0, r"H:\ai-learning\agent\project\assistant\tools")

import rag_tools

KNOWLEDGE_DIR = r"H:\ai-learning\agent\project\assistant\knowledge"
ok = rag_tools.init_rag(KNOWLEDGE_DIR)

print("init_rag 返回:", ok)
print("嵌入模型 _model 是否加载:", rag_tools._model is not None)
print("重排模型 _reranker 是否加载:", rag_tools._reranker is not None)
print("知识库 chunk 数量:", len(rag_tools._emb_data["chunks"]))
print(rag_tools.hybrid_search("注意力机制"))