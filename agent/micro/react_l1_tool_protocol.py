"""L0 — 纯对话循环（不接任何工具）

本关只有一个目标：搞清楚「多轮对话是怎么记住上下文的」。
这是 ReAct / Agent 的地基 —— 先把这个吃透，L1 再往里面加工具。

跑法：
    & E:\\deeplearning\\envs\\pytorch_env\\python.exe H:\\ai-learning\\agent\\micro\\react_l0_chat_loop.py
"""

import os
import sys
import json

import requests

# 让 common_config 能被 import（文件相对定位，不写死绝对路径，项目可移植）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common_config import API_KEY  # noqa: E402

API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

SYSTEM_PROMPT = "你是一个简洁的助手，回答控制在两句话以内。"


# =====================================================================
# 【参考卡片】调用 API 的"格式" —— 这部分属于查手册，已给全。
# 逐行读注释，不要跳；你要能说清每一行在干嘛。
# =====================================================================
def chat_once(messages):
    """向模型发【一次】请求，返回模型回复的纯文本 (str)。

    messages 是一个 list，每个元素是一个 dict，形如：
        [{"role": "system",    "content": "你是..."},
         {"role": "user",      "content": "你好"},
         {"role": "assistant", "content": "你好呀"}]
    """
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",  # 身份：谁在调用
            "Content-Type": "application/json",  # 我发过去的是 JSON
        },
        json={  # 请求体
            "model": MODEL,
            "messages": messages,  # ★ 把"全部历史"发过去
             "tools": TOOL_SCHEMAS,
            # "stream": True,      # 流式是 L3 的事，L0 不用
        },
    )
    data = resp.json()
    msg = data["choices"][0]["message"]

    return msg

def main():
    messages = []
    messages.append({"role":"system","content":SYSTEM_PROMPT})
    print("（输入 q 退出）")
    max_turn = 10
    while True:
        turn = 0
        user_input = input("\n你: ")
        if user_input.strip().lower() == "q":
            break
        messages.append({"role":"user","content":user_input})
        while True:
            if turn > max_turn:
                print("超过循环次数")
                break
            reply = chat_once(messages)
            if reply.get("tool_calls") == None:
                print(f"AI: {reply['content']}")
                break
            messages.append(reply)
            for tc in reply["tool_calls"]:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                result = TOOL_FUNCS[name](**args)
                messages.append({                         
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result),
                })
            turn += 1

def get_weather(city):
    return f"{city}的温度是23摄氏度，天气晴"

def calculator(operation,a,b):
    if operation == "add":
        return a+b
    elif operation == "multiply":
        return a*b
TOOL_FUNCS={
    "get_weather":get_weather,
    "calculator":calculator,
}
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "加法和乘法",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                    "type": "string",
                    "enum": ["add", "multiply"],   
                    },
                    "a": {"type": "number"},      
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        },
    }
]

if __name__ == "__main__":
    main()