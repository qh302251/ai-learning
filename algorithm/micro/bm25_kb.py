import json
from bm25_corpus import build_stats,bm25_score
from bm25_tokenize import tokenize_jieba

KB = r"H:\ai-learning\agent\project\assistant\knowledge\embeddings.json"

def load_chunks(path):
    """读 embeddings.json，返回 chunks 列表"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def bm25_rank(query, docs,stats):
    """返回按BM25分降序排好的[(索引，分数),...]"""
    qtokens = tokenize_jieba(query)
    pairs = []
    for i in range(len(docs)):
        s = bm25_score(docs[i],qtokens,stats)
        pairs.append((i,s))
    pairs.sort(key=lambda x : x[1],reverse=True)
    return pairs
def main():
    chunks = load_chunks(KB)

    # ① 原材料：每个 chunk 的 text → token 列表
    docs = []
    for c in chunks:
        docs.append(tokenize_jieba(c["text"]))      # ← 填：把 c["text"] 变成 token 列表

    # ② 建索引
    stats = build_stats(docs)
    N, doc_lens, avg_len, df_dict = stats
    print(f"N = {N},  avg_len = {avg_len:.1f},  词表大小 = {len(df_dict)}")

    # ③ 查询
    queries = [
        "神经网络是什么",
        "卷积神经网络处理什么数据",
        "assumed stress hybrid finite element",
    ]
    for q in queries:
        print(f"\n===== query: {q} =====")
        for i, s in bm25_rank(q, docs, stats)[:3]:
            c = chunks[i]
            print(f"[{s:+.3f}] {c['source']} #{c['chunk_index']}  {c['text'][:50]!r}")


if __name__ == "__main__":
    main()