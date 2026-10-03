"""RAG 工具 — 关键词搜索 + 语义搜索"""

import os
import re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from functools import partial

import numpy as np
from sentence_transformers import SentenceTransformer,CrossEncoder
from .bm25_tools import build_stats, tokenize_jieba, bm25_rank

_model = None
_emb_data = None
_doc_embeddings = None
_reranker = None
_bm25_docs = None      # 38 个 chunk 的 token 列表（建一次）
_bm25_stats = None     # BM25 索引统计 (N, doc_lens, avg_len, df_dict)

def init_rag(
    knowledge_dir: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
):
    """初始化语义搜索模型和向量数据"""
    global _model, _emb_data, _doc_embeddings, _reranker, _bm25_docs, _bm25_stats
    _model = SentenceTransformer(model_name)
    _reranker=CrossEncoder("BAAI/bge-reranker-v2-m3")
    emb_path = os.path.join(knowledge_dir, "embeddings.json")
    if os.path.exists(emb_path):
        with open(emb_path, "r", encoding="utf-8") as f:
            _emb_data = json.load(f)
        _doc_embeddings = np.array(_emb_data["embeddings"])
        # 建 BM25 索引（只建一次）
        _bm25_docs = []
        for c in _emb_data["chunks"]:
            _bm25_docs.append(tokenize_jieba(c["text"]))      # ← 填：一个 chunk 的 text 怎么变词表
        _bm25_stats = build_stats(_bm25_docs)
        return True
    return False


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_documents(keyword: str, knowledge_dir: str = None) -> str:
    """在知识库中搜索包含关键词的文档"""
    try:
        txt_files = [f for f in os.listdir(knowledge_dir) if f.endswith(".txt")]
        results = []
        for file_name in txt_files:
            file_path = os.path.join(knowledge_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if keyword in content:
                results.append(f"【{file_name}】\n{content}")
        if results:
            return "\n---\n".join(results)
        return f"未找到包含 '{keyword}' 的文档"
    except Exception as e:
        return f"搜索知识库时出错: {e}"

def hybrid_search(query: str, top_n: int = 5) -> str:
    if _model is None or _doc_embeddings is None:
        return "语义搜索模型未就绪，请先构建向量索引"
    query_vec = _model.encode(query)
    # ① BM25 关键词路：返回按 BM25 分降序的 chunk 索引列表
    def keyword_rank(query):
        paris=[]
        pairs = bm25_rank(query, _bm25_docs, _bm25_stats)
        return [ i for i, _ in pairs]        # ← 填：我们只要索引 i
    # ② 向量路：cosine_similarity(query_vec, _doc_embeddings[i]) 排序 → 得到 order_vec(索引列表)
    def vector_rank(query_vec,_doc_embeddings):
        pairs=[]
        for i in range(len(_doc_embeddings)):
            score=cosine_similarity(query_vec,_doc_embeddings[i])
            pairs.append((i,score))
        pairs.sort(key=lambda x : x[1],reverse=True)
        return [n for n,_ in pairs]
    # ③ rrf_fuse([order_kw, order_vec]) → 得到 [(索引, 融合分), ...]
    def rrf_fuse(ranked_lists,k=30):
        score={}
        for ranks in ranked_lists:
            for rank,i in enumerate(ranks):
                score[i]=score.get(i,0)+1/(rank+k)
        return sorted(score.items(),key=lambda x :x[1],reverse=True)
    # ④ 取前 top_k，用 _emb_data["chunks"][索引] 拼成文本返回
    rank_kw=keyword_rank(query)
    rank_vec=vector_rank(query_vec,_doc_embeddings)
    rrf_fuse_score=rrf_fuse([rank_kw,rank_vec])
    #召回top_k=20个切片
    top_k=20
    pool=rrf_fuse_score[:top_k]
    topn=rerank(query,pool,top_n)
    parts=[]
    for idx,score in topn:
        chunk = _emb_data["chunks"][idx]
        parts.append(f"【{chunk['source']} 第{chunk['chunk_index'] + 1}段】\n{chunk['text']}")
    return "\n---\n".join(parts)

def rerank(query, pool, top_n=5):
    """用 cross-encoder 对召回池精排，返回重排后的 top_n 个 chunk 索引。

    pool: 召回池，rrf_fuse 的结果 [(索引, 融合分), ...]
    top_n: 最终要返回几个
    返回: 重排后 top_n 个 chunk 索引列表（不是文本）
    """
    if _reranker is None:                       # 防御式：没加载就退回原序
        return pool[:top_n]

    indices = [idx for idx, _ in pool]          # 池里每个 chunk 的索引（顺序已知）
    texts = [_emb_data["chunks"][idx]["text"] for idx in indices]   # 对应文本

    # ① 造对子：每个候选配上一个 (query, 文本)
    cross=[(query,t) for t in texts]
    # ② _reranker.predict(...) 打分（返回的数组顺序 = 你喂的顺序）
    scores=_reranker.predict(cross)
    # ③ 把"索引"和"分数"配对、按分数降序排
    pairs=[]
    for i in range(len(indices)):
        pairs.append((indices[i],scores[i]))
    # ④ 取前 top_n 个索引返回
    return sorted(pairs,key=lambda x : x[1],reverse=True)[:top_n]



def register_all(registry, knowledge_dir):
    registry.register(
        "search_documents",
        partial(search_documents, knowledge_dir=knowledge_dir),
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "在本地知识库中搜索包含完整关键词的文本文件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "要搜索的关键词"},
                    },
                    "required": ["keyword"],
                },
            },
        },
    )
    registry.register(
        "hybrid_search",
        hybrid_search,
        {
            "type": "function",
            "function": {
                "name": "hybrid_search",
                "description": "在知识库中寻找与用户查询关键字匹配度和语义匹配度综合最高的几篇文档",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "用户的查询内容"},
                        "top_n": {"type": "integer", "description": "最终取出的文档切片数量"},
                    },
                    "required": ["query"],
                },
            },
        },
    )
