"""
v1 — 最小 ReAct Agent（基础版）
特点：一次性问答，无流式，固定测试用例
"""

import json
import requests

# ============ 配置 ============
import sys
sys.path.insert(0, r"H:\ai-learning")
from common_config import API_KEY
MODEL = "deepseek-chat"
API_URL = "https://api.deepseek.com/v1/chat/completions"


# ============ 工具定义 ============
def calculator(operation: str, a: float, b: float) -> float:
    """简单计算器"""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else "除数不能为0"
    elif operation == "subtract":
        return a - b
    return "不支持的运算"


def get_weather(city: str) -> str:
    """模拟查天气"""
    weather_data = {"北京": "晴, 28°C", "上海": "多云, 25°C", "成都": "阴, 22°C"}
    return weather_data.get(city, f"没有{city}的天气数据")


def read_file(file_path: str) -> str:
    """读取本地文本文件的内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, content: str) -> str:
    """将文本内容写入文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"文件{file_path}写入成功"


# 工具注册表
TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "read_file": read_file,
    "write_file": write_file,
}

# 工具的 JSON Schema
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行四则运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        },
    },
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
            "name": "read_file",
            "description": "读取本地文本文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写进本地文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件地址"},
                    "content": {"type": "string", "description": "写入内容"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
]

# ============ Agent 核心循环 ============
SYSTEM_PROMPT = (
    "你是一个智能助手，可以使用工具来帮助用户。如果需要工具就调用工具，否则直接回答。"
)


def run_agent(user_message: str, max_turns: int = 10) -> str:
    """一次性问答：传入用户消息，返回最终回答"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for turn in range(max_turns):
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": MODEL, "messages": messages, "tools": TOOL_SCHEMAS},
        )
        data = response.json()
        msg = data["choices"][0]["message"]
        messages.append(msg)

        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                result = TOOLS[func_name](**args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    }
                )
            continue
        else:
            return msg["content"]

    return "超过最大轮数"


# ============ 测试 ============
if __name__ == "__main__":
    result = run_agent("计算 123 × 456 等于多少？")
    print(f"Agent: {result}")

    result = run_agent("北京今天天气怎么样？")
    print(f"Agent: {result}")

    result = run_agent("读取test.txt文件的内容")
    print(f"Agent: {result}")

    result = run_agent("把'你好世界'写进hello_world.txt文件里面")
    print(f"Agent: {result}")
