import requests
import sys
sys.path.insert(0, r"H:\ai-learning")
from common_config import API_KEY
API_URL="https://api.deepseek.com/v1/chat/completions"

# 对比案例：问题 + 原始模型回答 + 微调模型回答
cases = [
    {
        "question": "请用鲁迅风格写一句鼓励的话",
        "original": "“世事如棋局，风云变幻中，唯有心静者能胜。愿你我皆是棋手，于纷扰之中寻得一方宁静之地，以坚韧之心，书写属于自己的传奇。”",
        "finetuned": "在生命的长河中，我们每个人都是独一无二的。不要因为一时的挫折而自暴自弃，也不要因为短暂的成功而沾沾自喜。只有不断努力、坚持不懈，才能绽放出属于自己的光芒。",
    },
    {
        "question": "什么是机器学习？请用通俗的语言解释",
        "original": "机器学习是一种人工智能技术，它让计算机能够从数据中自动学习和改进，而不需要进行明确的编程。简单来说，就是让计算机通过经验来“学习”如何做某件事，而不是直接被告诉怎么做。",
        "finetuned": "机器学习是一种人工智能技术，它可以让计算机通过数据和算法来不断学习并改进自己的性能。简单来说，就是让计算机在没有明确编程的情况下，能够从大量数据中自动发现规律，并据此做出预测或决策。",
    },
    {
        "question": "如何制定一个有效的学习计划？",
        "original": "1. 设定目标：首先，你需要设定明确的学习目标。这些目标应该具体、可衡量、可实现、相关性强和有时间限制的。2.制定计划：根据你的目标，制定一个详细的计划。3.分配资源：确定你将使用哪些资源来达到你的目标。",
        "finetuned": "要制定一个有效的学习计划，您可以遵循以下步骤：1.设定目标：首先确定您希望达到的目标，这些目标应该具体、可衡量、可实现、相关性强和时限性。2.分析资源：了解自己拥有的时间和金钱资源，并根据您的目标来安排时间。3.制定时间表：将目标分解为一系列任务，并安排完成时间。",
    },
]

results = []  # 存每个问题的 (A分, B分)

for case in cases:
    prompt =f"""
    你是一个严格的裁判。下面是两个模型对同一个问题的回答，请给它们打分。

    【问题】
    {case['question']}

    【回答A】（原始模型）
    {case['original']}

    【回答B】（微调后模型）
    {case['finetuned']}

    请从以下三个维度给 A 和 B 分别打分（每项 1-5 分，5 为最好）：
    1. 有用性（更切题、更有帮助）
    2. 文采（更有感染力）
    3. 格式（更自然、更像人话）

    只输出两行，不要输出其他内容：
    A:（A的总分，1-15 的整数）
    B:（B的总分，1-15 的整数）
    """

    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    content = resp.json()["choices"][0]["message"]["content"]
    print("=" * 40)
    print(f"问题：{case['question']}")
    print(content)

    # 解析裁判输出的 A:x / B:y
    a_score = b_score = 0
    for line in content.strip().split("\n"):
        if line.strip().startswith("A:"):
            a_score = int(line.split(":")[1].strip())
        elif line.strip().startswith("B:"):
            b_score = int(line.split(":")[1].strip())
    results.append((a_score, b_score))

print("=" * 40)
print("对比完成")

# 汇总
total_a = sum(r[0] for r in results)
total_b = sum(r[1] for r in results)
print("=" * 40)
print(f"总分对比：原始模型 {total_a} 分 vs 微调模型 {total_b} 分")