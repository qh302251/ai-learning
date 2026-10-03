import numpy as np

docs={
    "doc1":np.array([1.0,0.0]),
    "doc2":np.array([0.9,0.1]),
    "doc3":np.array([0.1,0.9]),
    "doc4":np.array([0.0,1.0]),
}

query=np.array([1.0,0.0])

def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def topk(query,docs,k):
    pairs=[]
    for name,vec in docs.items():
        score=cosine(query,vec)
        pairs.append((name,score))
    pairs.sort(key=lambda x : x[1],reverse=True)
    return pairs[:k]

print(topk(query,docs,2))