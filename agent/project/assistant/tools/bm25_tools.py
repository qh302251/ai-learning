"""BM25 关键词检索工具（jieba 分词）"""

import math
import re
import jieba

jieba.setLogLevel(20)      # ← 消掉 "Building prefix dict" 那一堆启动日志

def tokenize_jieba(text):
    tokens=[]
    for w in jieba.cut(text.lower()):
        if re.search(r'\w',w):
            tokens.append(w)
    return tokens

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

def bm25_rank(query, docs, stats):
    """返回按BM25分降序排好的[(索引，分数),...]"""
    qtokens = tokenize_jieba(query)
    pairs = []
    for i in range(len(docs)):
        s = bm25_score(docs[i],qtokens,stats)
        pairs.append((i,s))
    pairs.sort(key=lambda x : x[1],reverse=True)
    return pairs
