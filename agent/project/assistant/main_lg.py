"""个人知识库助手 — CLI 入口（LangGraph 版）

装配逻辑全在 bootstrap.build_agent()，本文件只负责交互循环。
"""

import time

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from agent_lg import stream_turn
from bootstrap import build_agent


def _print_events(events):
    """CLI 渲染：把事件流打成人类可读的输出，返回审批问题（None = 没中断）"""
    interrupted = None
    mid_line = False          # True = 光标停在行中间（正文刚输出过，还没换行）

    for kind, data in events:
        if kind == "text":
            print(data, end="", flush=True)
            mid_line = True

        elif kind == "tool_call":
            if mid_line:              # 正文没换行 → 先把这行收尾，再让提示独占一行
                print()
                mid_line = False
            print(f"  -> 调用 {data}")

        elif kind == "tool_done":
            if mid_line:
                print()
                mid_line = False
            print(f"  {data}")

        elif kind == "interrupt":
            interrupted = data

    print()
    return interrupted


def main():
    # ★ 进程级装配：模型 / 工具 / 图，只做一次
    app, mcp = build_agent()

    # ★ 会话级配置：每次启动都不同，不进 build_agent()
    thread_config = {
        "recursion_limit": 10,
        "configurable": {"thread_id": f"cli-{int(time.time())}"},
        "callbacks": [CallbackHandler()],
    }

    print("\n个人知识库助手已启动！输入 exit 退出\n")

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

        interrupted = _print_events(stream_turn(app, thread_config, user_text=user_input))
        while interrupted:
            answer = input(f"★ {interrupted} (y/n) ")
            interrupted = _print_events(stream_turn(app, thread_config, resume_value=answer))

    # 清理
    if mcp:
        mcp.stop()
    get_client().flush()
    print("再见！")


if __name__ == "__main__":
    main()