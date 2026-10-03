import torch
B, T, d_model = 2, 5, 8
n_head, d_head = 2, 4          # 2×4=8，正好=d_model

Q = torch.randn(B, T, d_model)  # (2, 5, 8)

# TODO 你要把 Q 变成 (2, 2, 5, 4) —— 即 (B, n_head, T, d_head)
Q_heads = Q.reshape(B,T,n_head,d_head).transpose(1,2)
print(Q_heads.shape)   # 期望 torch.Size([2, 2, 5, 4])


import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # ★ 这就是"被训练的权重" —— 4 个可学习投影
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, d = x.shape            # (B, T, d_model)
        Q = self.W_Q(x)              # (B, T, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

    # TODO A: 拆头 —— Q -> (B, n_heads, T, d_head)（K、V 同样）
        Q_h=Q.reshape(B,T,self.n_heads,self.d_head).transpose(1,2)
        K_h=K.reshape(B,T,self.n_heads,self.d_head).transpose(1,2)
        V_h=V.reshape(B,T,self.n_heads,self.d_head).transpose(1,2)
        
    # TODO C: 拼头 —— attention 结果 -> (B, T, d_model)
        result=self.attention(Q_h,K_h,V_h).transpose(1,2).reshape(B,T,self.d_model)
        # causal_result=self.causal_attention(Q_h,K_h,V_h).transpose(1,2).reshape(B,T,self.d_model)
        # TODO D: 过 self.W_O 输出
        out =self.W_O(result) 
        # out =self.W_O(causal_result) 
        return out
    
    # TODO B: 逐头 attention —— 复用你写好的 attention(Q,K,V)
    #         （4 维也能跑，因为它只对末尾两维做点积+softmax）
    def attention(self, Q, K, V):
        """无掩码（双向）：词1 能看到整句。"""
        d = K.size(-1)
        score = Q @ K.transpose(-2, -1) / (d ** 0.5)
        w = torch.softmax(score, dim=-1)
        return w @ V

    def causal_attention(self, Q, K, V):
        """有掩码（因果）：只能看前面的词。"""
        d = K.size(-1)
        score = Q @ K.transpose(-2, -1) / (d ** 0.5)
        L = Q.size(-2)
        mask = torch.tril(torch.ones(L,L,dtype=torch.bool))
        score = score.masked_fill(~mask,float('-inf'))
        w = torch.softmax(score, dim=-1)
        return w @ V

if __name__=="__main__":
    c=MultiHeadAttention(8,2)
    query=torch.randn(B,T,d_model)
    out=c.forward(query)
    print(out.shape)
    print(out)
    # 证明"每个头在看不同的东西" 
    Qm = c.W_Q(query).reshape(B, T, c.n_heads, c.d_head).transpose(1, 2)
    Km = c.W_K(query).reshape(B, T, c.n_heads, c.d_head).transpose(1, 2)
    Vm = c.W_V(query).reshape(B, T, c.n_heads, c.d_head).transpose(1, 2)
    w = torch.softmax(Qm @ Km.transpose(-2, -1) / (c.d_head ** 0.5), dim=-1)
    print("head0 第0行:", torch.round(w[0, 0, 0], decimals=3))
    print("head1 第0行:", torch.round(w[0, 1, 0], decimals=3))
    print("两头的第0行不一样吗?", (w[0,0,0] != w[0,1,0]).any().item())