import requests
import sys
sys.path.insert(0, r"H:\ai-learning")
from common_config import API_KEY
API_URL="https://api.deepseek.com/v1/chat/completions"


question="""一个水池，单开进水管 6 小时注满，单开排水管 8 小时排空。如果先开进水管 2 小时，再同时开排水管，还要几小时注满？只输出数字。"""
answers=[]
for i in range(5):
    resp=requests.post(
        API_URL,
        headers={
            "Authorization":f"Bearer {API_KEY}",
            "Content-Type":"application/json",
        },
        json={
            "model":"deepseek-chat",
            "messages":[{"role":"user","content":question}],
            "temperature":0.8,
        },
        timeout=30,
    )
    answer=resp.json()["choices"][0]["message"]["content"]
    answers.append(answer)
    print(f"第{i+1}次:{answer}")
from collections import Counter
count=Counter(answers)
print(f"\n最常见的答案:{count.most_common(1)[0][0]}")