# -*- coding: utf-8 -*-
"""RAGAS golden 测试集 —— 金标准（问题 + 标准答案 + 应命中 chunk 的独特子串）

字段说明：
  question      用户问题
  ground_truth  一句话标准答案（回答正确性判定用）
  gold          应命中 chunk 里的"独特子串"（判召回/命中用，须能在 chunks[].text 里找到）
  source        来自哪个文件（方便核对）
  chunk_idx     应命中的 chunk 索引（_emb_data['chunks'] 的下标）

⚠️ 评估时注意：论文 chunk 内含换行（chunks[].text 有 "\n"），
   比较 gold 前要把两侧的空白规整成单空格（re.sub(r'\s+',' ', ...)），否则会因换行判不命中。
"""

golden = [
    # ============ 5 个主题短文（关键词/中文都能正中） ============
    {
        "question": "卷积神经网络专门用于处理什么类型的数据？",
        "ground_truth": "网格状数据，如图像。",
        "gold": "卷积神经网络（CNN）专门用于处理网格状数据，如图像",
        "source": "cnn.txt",
        "chunk_idx": [0],
    },
    {
        "question": "神经网络由哪些部分组成？",
        "ground_truth": "输入层、隐藏层和输出层。",
        "gold": "神经网络是模拟人脑神经元结构的计算模型，由输入层、隐藏层和输出层组成",
        "source": "nn.txt",
        "chunk_idx": [1],
    },
    {
        "question": "BERT 和 GPT 都是基于什么架构？",
        "ground_truth": "Transformer 架构（基于自注意力机制）。",
        "gold": "Transformer架构基于自注意力机制，彻底改变了NLP领域。BERT和GPT都是基于Transformer",
        "source": "transformer.txt",
        "chunk_idx": [2],
    },
    {
        "question": "强化学习的目标是什么？",
        "ground_truth": "通过与智能体环境交互，最大化累积奖励；Q-Learning 是经典算法之一。",
        "gold": "强化学习通过智能体与环境交互，以最大化累积奖励为目标",
        "source": "rl.txt",
        "chunk_idx": [3],
    },
    {
        "question": "循环神经网络（RNN）适用于什么数据？LSTM 是什么？",
        "ground_truth": "适用于序列数据（如文本、时间序列）；LSTM 是 RNN 的一种改进变体。",
        "gold": "循环神经网络（RNN）适用于序列数据，如文本和时间序列",
        "source": "rnn.txt",
        "chunk_idx": [4],
    },
    # ============ 5 个论文（long.txt，中文问英文论文，测语义/跨语种） ============
    {
        "question": "这篇论文研究的是哪一类有限元方法？",
        "ground_truth": "假设应力杂交有限元方法（assumed stress hybrid finite element methods）。",
        "gold": "Assumed stress hybrid methods are known to improve the performance",
        "source": "long.txt",
        "chunk_idx": [7],
    },
    {
        "question": "论文所研究的杂交方法基于哪个变分原理？",
        "ground_truth": "Hellinger-Reissner 变分原理。",
        "gold": "Reissner variational principle",
        "source": "long.txt",
        "chunk_idx": [8],
    },
    {
        "question": "论文分析的两个 4 节点杂交应力四边形单元分别由谁提出？",
        "ground_truth": "PS 有限元（Pian 和 Sumihara）和 ECQ4 有限元（Xie 和 Zhou）。",
        "gold": "the ECQ4 finite element by Xie and Zhou",
        "source": "long.txt",
        "chunk_idx": [30],
    },
    {
        "question": "论文证明这两个方案不存在哪种锁死现象？",
        "ground_truth": "泊松锁死（Poisson-locking），误差上界与 Lamé 常数 k 无关。",
        "gold": "the two schemes are free from Poisson-locking",
        "source": "long.txt",
        "chunk_idx": [11],
    },
    {
        "question": "论文推导出了针对什么的残差型后验误差估计？",
        "ground_truth": "针对应力（L2 范数）和位移（H1 范数）的残差型后验误差估计。",
        "gold": "residual-based a posteriori error estimators for the stress in L2",
        "source": "long.txt",
        "chunk_idx": [12],
    },
]