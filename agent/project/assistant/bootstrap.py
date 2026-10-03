"""启动装配：把 Agent 搭起来（CLI 和 HTTP 服务共用）

★ 这个模块很慢：加载 embedding + reranker 模型、启动 MCP 子进程。
   一个进程只能调用 build_agent() 一次，不要在请求处理里调用它。
"""

import os

from config import (
    API_KEY,
    API_URL,
    MODEL,
    SYSTEM_PROMPT,
    KNOWLEDGE_DIR,
    MCP_SCRIPT,
)
from agent_lg import build_app
from mcp_client import MCPClient
from tool_registry import ToolRegistry
from tools import local_tools, memory_tools, rag_tools


def build_agent():
    """构建 Agent，返回 (app, mcp)。

    mcp 是 MCP 子进程的句柄，**调用方负责在退出时 mcp.stop()**；
    为 None 表示没启动（脚本不存在或启动失败）。
    """
    # ★ 快速失败：与其让人对着一个 401 发呆，不如在这里把原因说清楚
    if not API_KEY:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY。\n"
            "  本地运行：在仓库根 .env 里配置 DEEPSEEK_API_KEY=sk-xxx\n"
            "  容器运行：docker run -e DEEPSEEK_API_KEY=sk-xxx ... 或用 compose 的 env_file"
        )

    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

    # 1. 工具注册表
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

    # 5. 启动 MCP（可选，脚本不存在则跳过）
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

    # 6. 建图（把上面装配好的东西交给 LangGraph）
    #
    # ★ 不再在这里把记忆拼进 system prompt！
    #   记忆是【请求级】数据（每个 user 不同、而且随时会变），启动时读一次会造成两个问题：
    #     ① 跨用户泄露 —— bob 能看到 alice 的档案（已实测到）
    #     ② 新记忆不生效 —— remember 之后要重启容器才看得见
    #   改成注入 memory_provider，由 llm_node 按 config 里的 user_id 现取。
    print(f"记忆库就绪（按用户隔离）: {memory_tools.stats()}")

    api_config = {"api_key": API_KEY, "api_url": API_URL, "model": MODEL}
    app = build_app(
        registry,
        api_config,
        system_prompt=SYSTEM_PROMPT,
        memory_provider=memory_tools.render_for,
    )

    print(f"工具就绪: {', '.join(registry.list_tools())}")
    return app, mcp