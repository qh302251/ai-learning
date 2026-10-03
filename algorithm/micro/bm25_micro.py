import math

def bm25_term(tf, df, N, doc_len, avg_len, k1=1.2, b=0.75):
    """BM25 中，查询词 t 对一篇文档的得分。
    tf      : 词 t 在本文档出现次数
    df      : 包含词 t 的文档数（document frequency）
    N       : 文档总数
    doc_len : 本文档长度（词数）
    avg_len : 全库平均文档长度
    """
    # === 你写这里 ===
    # 1) IDF = log((N - df + 0.5) / (df + 0.5))
    IDF = math.log((N-df+0.5)/(df+0.5))
    # 2) TF部分 = tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))
    TF = tf*(k1+1)/(tf+k1*(1-b+b*(doc_len/avg_len)))
    # 3) return IDF * TF部分
    return IDF * TF