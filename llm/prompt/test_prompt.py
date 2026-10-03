import requests
import sys
sys.path.insert(0, r"H:\ai-learning")
from common_config import API_KEY
API_URL="https://api.deepseek.com/v1/chat/completions"

resp= requests.post(
    API_URL,
    headers={
        "Authorization":f"Bearer {API_KEY}",
        "Content-Type":"application/json",
    },
    json={
        "model":"deepseek-chat",
        "messages":[
            {"role":"user","content":"""分类下面的评论，格式如下:
            情感：×××
            理由：×××
            例子1：
            评论：这家餐厅的菜太好吃了，下次还要来
            情感：正面
            理由：菜品满意，有复购意愿
            
            例子2：
            评论：上菜太慢了，等了整整一个小时
            情感：负面
            理由：等待时间过长
              
            例子3：
            评论：环境还行，价格也适中，没什么特别的感觉
            情感：中性
            理由：各方面都普通，没有突出的好或差
            
            现在分类：
            评论：价格不算贵，但质量也就那样
            """
            }
        ],
    },
    timeout=30,
)
print(resp.json()["choices"][0]["message"]["content"])