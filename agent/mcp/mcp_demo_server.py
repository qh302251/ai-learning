"""
MCP Server 最小实现 — 通过 stdin/stdout 通信
"""

import json
import sys


def calculator(operation, a, b):
    """执行四则运算"""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else "除数不能为0"
    elif operation == "subtract":
        return a - b
    return "不支持的运算"


# 工具列表（MCP 格式）
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
    }
]


while True:
    # 从 stdin 读一行 JSON
    line = sys.stdin.readline()
    if not line:
        break

    req = json.loads(line.strip())
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    # 处理 tools/list
    if method == "tools/list":
        resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    # 处理 tools/call
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "calculator":
            result = calculator(**args)
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": str(result)}]},
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "工具不存在"},
            }

    # 写入 stdout
    print(json.dumps(resp, ensure_ascii=False), flush=True)
