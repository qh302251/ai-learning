"""
最小 ReAct Agent — 不依赖任何框架，50行实现核心循环
"""

import json
import requests

# ============ 配置 ============
import sys
sys.path.insert(0, r"H:\ai-learning")
from common_config import API_KEY
MODEL = "deepseek-chat"  # 或其他模型
API_URL = "https://api.deepseek.com/v1/chat/completions"  # 改成对应 API 地址


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


# 工具注册表 — dispatch 表，不用 if-elif 长链
TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "read_file": read_file,
    "write_file": write_file,
}

# 工具的 JSON Schema，告诉 LLM 有哪些工具可用
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
SYSTEM_PROMPT = """
你是一个使用工具的助手。请根据工具返回的结果自然地回答用户。下面是一些示例，请严格遵循它们的回复方式。
如果你能正确回答用户的问题，记得在末尾加上嘲讽用户的话
记住：你的工具就是你的全部知识，你没有其他的知识来源
用户：请计算sqrt(3)
助手：[调用 calculator 工具]
工具返回：不支持的运算
助手：抱歉，我现在还不会这个运算
"""


def run_agent(messages: list, max_turns: int = 10) -> str:
    """ReAct 循环：处理 tool calls，直到 LLM 给出最终答案（流式输出）"""
    for turn in range(max_turns):
        # 1. 流式调用 LLM
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "stream": True,
            },
            stream=True,
        )

        # 逐块累加流式数据
        full_content = ""
        tool_calls_buf = {}  # {index: {id, function: {name, arguments}}}
        times = 0
        content_chunks = 0
        tool_chunks = 0
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            chunk = json.loads(data_str)
            times += 1
            delta = chunk["choices"][0]["delta"]

            # 内容片段 → 立即打印
            if delta.get("content"):
                full_content += delta["content"]
                content_chunks += 1
                print(delta["content"], end="", flush=True)

            # 工具调用片段 → 累加到缓冲区
            if "tool_calls" in delta:
                tool_chunks += 1
                for tc in delta["tool_calls"]:
                    idx = tc["index"]
                    if idx not in tool_calls_buf:
                        tool_calls_buf[idx] = {
                            "id": "",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.get("id"):
                        tool_calls_buf[idx]["id"] = tc["id"]
                    if "function" in tc:
                        if tc["function"].get("name"):
                            tool_calls_buf[idx]["function"]["name"] += tc["function"][
                                "name"
                            ]
                        if tc["function"].get("arguments"):
                            tool_calls_buf[idx]["function"]["arguments"] += tc[
                                "function"
                            ]["arguments"]

        print(f"\n(一共调用了{content_chunks}次文本块)")
        print(f"\n(一共出现了{tool_chunks}次工具调用delta块)")
        print(f"\n(一共接收了{times}个数据块)")  # 流式结束，换行

        # 2. 检查是否有工具调用
        if tool_calls_buf:
            # 构造 assistant 消息（包含 tool_calls）
            tool_calls_list = [
                {
                    "id": tool_calls_buf[idx]["id"],
                    "type": "function",
                    "function": {
                        "name": tool_calls_buf[idx]["function"]["name"],
                        "arguments": tool_calls_buf[idx]["function"]["arguments"],
                    },
                }
                for idx in sorted(tool_calls_buf.keys())
            ]
            assistant_msg = {"role": "assistant", "content": full_content or None}
            assistant_msg["tool_calls"] = tool_calls_list
            messages.append(assistant_msg)

            # 执行每个工具
            for tc_info in tool_calls_list:
                func_name = tc_info["function"]["name"]
                args = json.loads(tc_info["function"]["arguments"])
                print(f"  → 调用 {func_name}({args})")
                result = TOOLS[func_name](**args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_info["id"],
                        "content": str(result),
                    }
                )
            continue

        # 没有工具调用 → 最终答案
        messages.append({"role": "assistant", "content": full_content})
        return full_content

    return "超过最大轮数"


# ============ 测试 ============
if __name__ == "__main__":
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("🤖 Agent 已启动！输入 exit 退出\n")

    while True:
        user_input = input(">>> ")
        if user_input.lower() == "exit":
            print("再见！")
            break

        messages.append({"role": "user", "content": user_input})
        run_agent(messages)
        print()  # 轮次之间空一行
