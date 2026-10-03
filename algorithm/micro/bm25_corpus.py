import math

def bm25_term(tf, df, N, doc_len, avg_len, k1=1.2, b=0.75):
    IDF = math.log((N-df+0.5)/(df+0.5))
    TF = tf*(k1+1)/(tf+k1*(1-b+b*(doc_len/avg_len)))
    return IDF * TF

def build_stats(docs):
    """docs: list，每项是一个 token 列表。
    返回 (N, doc_lens, avg_len, df_dict)
    """
    N=len(docs)
    doc_lens=[]
    total_lens=0
    df_dict={}
    for i in range(len(docs)):
        doc_lens.append(len(docs[i]))
        total_lens+=doc_lens[i]
        for w in set(docs[i]):
            df_dict[w]=df_dict.get(w,0)+1
    avg_len=total_lens/len(docs)
    return (N,doc_lens,avg_len,df_dict)


def bm25_score(doc_tokens, query_tokens, stats):
    """一篇文档对一个查询的 BM25 分 = 查询里每个词的 bm25_term 之和。"""
    N, doc_lens, avg_len, df_dict = stats
    score=0
    for i in query_tokens:
        tf=doc_tokens.count(i)
        if tf == 0:
            continue
        score+=bm25_term(tf,df_dict[i],N, len(doc_tokens), avg_len, k1=1.2, b=0.75)
    return score

if __name__ == "__main__":
    docs = [
        ["神经","网络","用于","图像"],
        ["神经","网络","用于","文本"],
        ["图像","识别","是","核心"],
    ]
    stats = build_stats(docs)
    print("stats:", stats)
    query = ["神经", "图像"]
    for i, d in enumerate(docs):
        print(f"D{i+1} score = {bm25_score(d, query, stats):.3f}")