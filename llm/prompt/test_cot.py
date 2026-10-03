import requests
import sys
sys.path.insert(0, r"H:\ai-learning")
from common_config import API_KEY
API_URL = "https://api.deepseek.com/v1/chat/completions"

resp = requests.post(
    API_URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": """一个商店进了一批苹果，第一天卖出总数的 1/3，第二天卖出剩下的 1/4，第三天卖出 15 个后还剩 30 个。请问这批苹果原来有多少个？"""}
        ],
    },
    timeout=30,
)

print(resp.json()["choices"][0]["message"]["content"])
