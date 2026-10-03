import requests
import sys
sys.path.insert(0, r"H:\ai-learning")
from common_config import API_KEY
API_URL="https://api.deepseek.com/v1/chat/completions"

resp=requests.post(
    API_URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":"application/json",
    },
    json={
        "model":"deepseek-chat",
        "messages":[
            # {"role":"system","content":"你是一个小学数学老师，用最简单易懂的方式解释概念，多用生活例子，少用术语。"},
            # {"role":"system","content":"你是一个算法工程师。"},
            {"role":"user","content":"什么是梯度下降？"}
        ],
    },
    timeout=30,
)

print(resp.json()["choices"][0]["message"]["content"])