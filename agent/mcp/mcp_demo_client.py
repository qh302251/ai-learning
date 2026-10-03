"""
MCP Client 最小实现 — 启动 Server 并通过 stdin/stdout 通信
"""

import subprocess
import json

proc = subprocess.Popen(
    ["python", "mcp_demo_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)

# 1. 发送 tools/list
req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
proc.stdin.write(json.dumps(req) + "\n")
proc.stdin.flush()

resp = proc.stdout.readline()
print("tools/list 响应:", resp)

# 2.发送一个 tools/call，算 1 + 2
req2 = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "calculator", "arguments": {"operation": "add", "a": 1, "b": 2}},
}
proc.stdin.write(json.dumps(req2) + "\n")
proc.stdin.flush()
resp2 = proc.stdout.readline()
print("tools/call 响应:", resp2)


# 3. 关闭
proc.stdin.close()
proc.wait()
