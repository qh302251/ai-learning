import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import CrossEncoder

# ① 加载重排模型（多语言，中文效果最好，约2.3GB；嫌大可换 BAAI/bge-reranker-base）
model = CrossEncoder("BAAI/bge-reranker-v2-m3")

query = "注意力机制"
candidates = [
    "Transformer架构基于自注意力机制，彻底改变了NLP领域。BERT和GPT都是基于Transformer。",
    "卷积神经网络（CNN）专门用于处理网格状数据，如图像。它的核心操作是卷积和池化。",
    "循环神经网络（RNN）适用于序列数据，如文本和时间序列。LSTM是RNN的一种改进变体。",
    "强化学习通过智能体与环境交互，以最大化累积奖励为目标。Q-Learning是经典算法之一。",
    "Finally, a b abbreviates a [ b [ a. We define two spaces as follows:",  # 乱码段
]

# ② 给每个候选打分
scores = model.predict([(query, c) for c in candidates])
print("形状:", scores.shape)     # 应该是 (5,)
print("分数:", scores)           # 5 个数字，transformer 那个应该最大、乱码段最小

# ③ 把 (索引, 分数) 存起来、按分数排序（你已经很熟的 pairs→sort 套路）
pairs=[(i,c) for i,c in enumerate(scores)]
pairs.sort(key=lambda x : x[1],reverse=True)
# ④ 打印 "分数从高到低" 的排名，看 cross-encoder 把谁排第一、谁垫底
print(pairs)