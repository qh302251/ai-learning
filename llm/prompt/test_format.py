import requests
import sys
sys.path.insert(0, r"H:\ai-learning")
from common_config import API_KEY
API_URL="https://api.deepseek.com/v1/chat/completions"

resp=requests.post(
    API_URL,
    headers={
        "Authorization":f"Bearer {API_KEY}",
        "Content-Type":"application/json",
    },
    json={
        "model":"deepseek-chat",
        "messages":[
            {"role":"user","content":"""从一下文本中提取信息，返回JSON(不要返回其他文字):
            文本:张三，28岁，北京大学计算机硕士，精通Python和Java，5年工作经验。
            JSON格式：
            {
                "name":"姓名"，
                "age":年龄，
                "education":"学历"，
                "skills":["技能1","技能2"]
            }"""}
        ],
    },
    timeout=30
)
print(resp.json()["choices"][0]["message"]["content"])