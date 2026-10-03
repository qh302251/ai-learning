"""
MCP Server — 托管所有通用工具
"""

import json
import sys

# 固定 stdin/stdout 为 UTF-8：否则会跟随系统默认编码(GBK)或 PYTHONIOENCODING，
# 一旦两端编码不一致就会报 "gbk codec can't decode byte 0xa1" 这种误导性错误。
sys.stdin.reconfigure(encoding="utf-8", errors="replace")
sys.stdout.reconfigure(encoding="utf-8")


def calculator(operation, a, b):
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else "除数不能为0"
    elif operation == "subtract":
        return a - b
    return "不支持的运算"


def get_weather(city):
    weather_data = {"北京": "晴, 28°C", "上海": "多云, 25°C", "成都": "阴, 22°C"}
    return weather_data.get(city, f"没有{city}的天气数据")


TOOLS = [
    {
        "name": "calculator",
        "description": "执行四则运算",
        "inputSchema": {
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
    {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
            },
            "required": ["city"],
        },
    },
]

# 工具分发表
_HANDLERS = {
    "calculator": calculator,
    "get_weather": get_weather,
}

while True:
    line = sys.stdin.readline()
    if not line:
        break

    try:
        req = json.loads(line.strip())
    except json.JSONDecodeError as e:
        continue

    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    try:
        if method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {})
            handler = _HANDLERS.get(name)
            if handler:
                result = handler(**args)
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": str(result)}]},
                }
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": f"未知工具: {name}"},
                }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "方法不存在"},
            }
    except Exception as e:
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": str(e)},
        }

    print(json.dumps(resp, ensure_ascii=False), flush=True)
