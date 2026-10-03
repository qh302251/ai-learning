"""个人知识库助手 — 入口"""

import os
import sys
from pathlib import Path

from agent_core import run_agent
from mcp_client import MCPClient
from tool_registry import ToolRegistry
from tools import local_tools, rag_tools, memory_tools


def _find_repo_root() -> Path:
    """向上查找含 common_config.py 的目录（common_config.py 放在仓库根）。

    不写死盘符和层级：换机器、挪目录、调整目录深度都不会 ImportError。
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "common_config.py").exists():
            return parent
    return Path(r"H:\ai-learning")  # 兜底：推导失败时保持原有行为

sys.path.insert(0, str(_find_repo_root()))
from common_config import API_KEY  # noqa: E402  必须先补好 sys.path 才能导入

# ============ 配置 ============
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
MCP_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")

SYSTEM_PROMPT = """你是一个个人知识库助手。你有以下能力：
1. 文件操作 — 读取/写入本地文件
2. 知识搜索 — 在本地知识库中搜索文档（关键词 + 语义）
3. 长期记忆 — 记住用户的事实和偏好
4. MCP 工具 — 通过 MCP 协议扩展更多能力

根据用户问题选择合适的工具，用工具结果回答。回答简洁自然。"""


def main():
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

    # 1. 初始化工具注册表
    registry = ToolRegistry()

    # 2. 注册本地工具
    local_tools.register_all(registry)

    # 3. 注册 RAG 工具（同时初始化模型）
    print("加载语义搜索模型...")
    rag_ready = rag_tools.init_rag(KNOWLEDGE_DIR)
    rag_tools.register_all(registry, KNOWLEDGE_DIR)
    print(
        "  语义搜索就绪"
        if rag_ready
        else "  语义搜索模型就绪（知识库为空，添加文档后运行 build_index.py 构建索引）"
    )

    # 4. 注册记忆工具
    memory_tools.register_all(registry)

    # 5. 启动 MCP（可选，文件不存在则跳过）
    mcp = None
    if os.path.exists(MCP_SCRIPT):
        print("启动 MCP Server...")
        try:
            mcp = MCPClient(MCP_SCRIPT)
            mcp.start()
            mcp_schemas = mcp.discover_tools()
            registry.register_mcp_tools(mcp_schemas, mcp.call_tool)
            print(f"  MCP 就绪，发现 {len(mcp_schemas)} 个远程工具")
        except Exception as e:
            print(f"  MCP 启动失败（不影响本地工具）: {e}")

    # 6. 加载记忆到 system prompt
    memories = memory_tools.get_all()
    if memories:
        memory_text = "已知信息：" + "; ".join(
            [f"{m['key']}={m['value']}" for m in memories]
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + memory_text}]
        print(f"已加载 {len(memories)} 条记忆")
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 7. 交互循环
    api_config = {"api_key": API_KEY, "api_url": API_URL, "model": MODEL}
    tools_list = ", ".join(registry.list_tools())
    print(f"\n个人知识库助手已启动！（当前工具: {tools_list}）")
    print("输入 exit 退出\n")

    while True:
        try:
            user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("exit", "quit"):
            break

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        run_agent(messages, registry, api_config)
        print()

    # 清理
    if mcp:
        mcp.stop()
    print("再见！")


if __name__ == "__main__":
    main()
