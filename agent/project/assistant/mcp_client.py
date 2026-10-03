"""MCP 客户端 — 进程管理与 JSON-RPC 通信"""

import json
import subprocess
import sys


class MCPClient:
    def __init__(self, server_script: str):
        self.server_script = server_script
        self._proc = None
        self._req_id = 0

    def start(self):
        # 用 sys.executable 而不是裸 "python"：裸名字要靠 PATH 解析，
        # PATH 上的 python 可能是失效的（如 Windows Store 占位存根），
        # 那样 MCP 子进程会起不来；用当前解释器可保证两端一致。
        self._proc = subprocess.Popen(
            [sys.executable, self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",   # 固定 UTF-8，不依赖系统默认编码(GBK)或 PYTHONIOENCODING
        )

    def _send(self, method, params=None):
        if self._proc is None:
            raise Exception("MCP Server 未启动")
        self._req_id += 1
        req = {"jsonrpc": "2.0", "id": self._req_id, "method": method}
        if params:
            req["params"] = params
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        resp = json.loads(self._proc.stdout.readline())
        if "error" in resp:
            raise Exception(resp["error"]["message"])
        return resp["result"]

    def discover_tools(self):
        """返回 OpenAI 兼容的 tool schema 列表"""
        result = self._send("tools/list")
        schemas = []
        for t in result["tools"]:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["inputSchema"],
                    },
                }
            )
        return schemas

    def call_tool(self, name, arguments):
        result = self._send("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content", [])
        return content[0]["text"] if content else ""

    def stop(self):
        if self._proc:
            self._proc.stdin.close()
            self._proc.wait()
            self._proc = None
