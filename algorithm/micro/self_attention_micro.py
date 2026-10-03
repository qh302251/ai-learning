import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

# def scaled_dot_attention(query, key, value):
#     """query/key/value: 形状都是 (序列长度, d_k)
#     返回: (序列长度, d_k) 的注意力输出
#     """
#     d_k = key.shape[-1]
#     # ① 算得分矩阵 scores = query @ key.T   → 形状 (seq, seq)
#     score=np.dot(query,key.T)
#     #    scores[i][j] = token i 对 token j 的"关注度"
#     # ② 缩放：除以 sqrt(d_k)
#     scaled_score=score/np.sqrt(d_k)
#     # ③ 对每一行做 softmax → 得到权重（每行和为 1）
#     weights=np.exp(scaled_score-np.max(scaled_score,axis=1,keepdims=True))
#     temp=weights/np.sum(weights,axis=1,keepdims=True)
#     output=np.dot(temp,value)
#     print("注意力权重:\n", np.round(temp, 3))
#     print("每行和:", temp.sum(axis=1))   # 应该全是 [1., 1., ...]
#     return output

# d_k=4
# X=np.array([
#     [1,0,0,2],
#     [0,1,0,1],
#     [0,0,1,0],
# ])

# out = scaled_dot_attention(X, X, X)   # 自注意力：Q=K=V=X（X 里已经带位置）


# def attention(Q,K,V):
#     d=K.shape[-1]
#     score=Q @ K.T/np.sqrt(d)
#     scaled_score=score-np.max(score,axis=1,keepdims=True)
#     weights=np.exp(scaled_score)/np.sum(np.exp(scaled_score),axis=1,keepdims=True)
#     output=weights @ V  
#     return output

# #带batch维度的numpy版本
# def attention(Q,K,V):
#     d=K.shape[-1]
#     score=Q @ np.swapaxes(K,-2,-1)/np.sqrt(d)
#     scaled_score=score-np.max(score,axis=-1,keepdims=True)
#     weights=np.exp(scaled_score)/np.sum(np.exp(scaled_score),axis=-1,keepdims=True)
#     output=weights @ V
#     return output

# def attention_torch(Q,K,V):
#     d=K.shape[-1]#或者Q.size(-1)
#     score=Q @ K.transpose(-2,-1)/d**0.5
#     weights=torch.softmax(score,dim=-1)
#     output=weights @ V
#     return output

# Q = torch.randn(2, 3, 4)
# K = torch.randn(2, 3, 4)
# V = torch.randn(2, 3, 4)
# out = attention_torch(Q, K, V)
# print(out.shape)      # 期望 (2, 3, 4)
# print(out)


import torch.nn.functional as F

torch.manual_seed(0)

def attention(Q, K, V):
    """无掩码（双向）：词1 能看到整句。"""
    d = K.size(-1)
    score = Q @ K.transpose(-2, -1) / (d ** 0.5)
    w = torch.softmax(score, dim=-1)
    return w, w @ V

def causal_attention(Q, K, V):
    """有掩码（因果）：只能看前面的词。"""
    d = K.size(-1)
    score = Q @ K.transpose(-2, -1) / (d ** 0.5)
    L = Q.size(-2)
    mask = torch.tril(torch.ones(L,L,dtype=torch.bool))
    score = score.masked_fill(~mask,float('-inf'))
    w = torch.softmax(score, dim=-1)
    return w, w @ V

# ---- 演示：print 出来对比 ----
Q = torch.randn(1, 4, 8)   # (batch=1, seq=4, d=8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)

w_no, _  = attention(Q, K, V)
w_cas, _ = causal_attention(Q, K, V)

print("无掩码（双向）weights:\n", w_no[0])
print("\n有掩码（因果）weights:\n", w_cas[0])
print("\n每行和（应都为 1）:")
print("无掩码:", w_no[0].sum(dim=-1))
print("有掩码:", w_cas[0].sum(dim=-1))

