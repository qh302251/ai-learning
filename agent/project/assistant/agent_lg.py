"""LangGraph 版 Agent —— 替代 agent_core.run_agent"""

import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command

from config import DB_PATH 


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_app(registry, api_config, system_prompt="", memory_provider=None):
    """memory_provider: 可调用对象，签名 (user_id) -> str，返回该用户的记忆文本块。

    ★ 为什么用"注入"而不是直接 import memory_tools：
      agent_lg 是【编排层】，不该知道记忆存在哪个文件、什么格式。
      注入之后 agent_lg 只依赖一个"给我一段文本"的约定 —— 以后换存储、换向量库，
      这个文件一行都不用改。（依赖倒置）
    """

    llm = ChatOpenAI(
        model=api_config.get("model", "deepseek-chat"),
        api_key=api_config["api_key"],
        base_url=api_config["api_url"].removesuffix("/chat/completions"),
    )
    llm_tools = llm.bind_tools(registry.get_all_schemas())

    def _user_id_of(config):
        """从本轮请求的 config 里取 user_id；取不到返回 None。"""
        return ((config or {}).get("configurable") or {}).get("user_id")

    def llm_node(state, config):
        # ★ LangGraph 会把本轮请求的 config 作为【第二个参数】传进来 ——
        #   这就是把"请求级"信息送进"进程级"闭包的官方通道，不需要任何全局变量。
        msgs = state["messages"]
        prompt = system_prompt

        # ★ 每次调用模型前都【按当前用户现取】记忆。
        #   不在启动时烤进 prompt —— 那既是跨用户泄露的源头，
        #   也让新记住的东西要等重启才生效。
        if memory_provider:
            uid = _user_id_of(config)
            if uid:
                block = memory_provider(uid)
                if block:
                    prompt = f"{prompt}\n\n{block}" if prompt else block

        if prompt:
            msgs = [SystemMessage(content=prompt)] + msgs   # ★ 只在发给模型时拼
        return {"messages": [llm_tools.invoke(msgs)]}

    def tools_node(state, config):
        # ★ 同一个通道，这里取出 user_id 交给 registry
        user_id = _user_id_of(config)
        last = state["messages"][-1]
        out = []
        for tc in last.tool_calls:
            if registry.needs_approval(tc["name"]):
                answer = interrupt(f"要执行 {tc['name']}({tc['args']})，批准吗？")
                if answer != "y":
                    out.append(ToolMessage(
                        content="【人工拒绝】用户未批准该操作。",
                        tool_call_id=tc["id"],
                    ))
                    continue
            try:
                result = registry.dispatch(
                    tc["name"],
                    context={"user_id": user_id},   # ★ 请求级上下文
                    **tc["args"],
                )
            except Exception as e:
                result = f"工具执行出错: {e}"
            out.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        return {"messages": out}

    def should_continue(state):
        last = state["messages"][-1]
        if last.tool_calls:
            return "tools"
        return END

    g = StateGraph(State)
    g.add_edge(START, "llm")
    g.add_node("llm", llm_node)
    g.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    g.add_node("tools", tools_node)
    g.add_edge("tools", "llm")

    # ★ 单进程假设（重要，别不知不觉破坏它）：
    #   SqliteSaver 内部有一把 threading.Lock，把【所有】DB 访问串行化了 ——
    #   所以同一个进程内多线程并发是安全的，不会 database is locked。
    #   但这把锁是【进程内】的：
    #     · uvicorn --workers N
    #     · docker compose --scale assistant=N
    #     · 外部脚本/备份工具同时写同一个 db
    #   这三种情况都会跨进程撞 SQLite 的文件锁 → database is locked。
    #   真要横向扩展，请换 Postgres checkpointer，不要靠调大 timeout 硬扛。
    #
    # timeout=30：这是 SQLite 的 busy timeout —— 拿不到写锁时【愿意等 30 秒】。
    #   （Python 默认 5 秒；调大是给多进程场景兜底，不是修 bug）
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30.0)
    return g.compile(checkpointer=SqliteSaver(conn))


def stream_turn(app, config, user_text=None, resume_value=None):
    """把一轮对话拆成事件流（生成器）。

    yield 的事件（元组：(类型, 数据)）：
      ("text",      str)  —— 模型输出的一段文字（token 级）
      ("tool_call", str)  —— 模型申请调用工具，如 "hybrid_search({'query':...})"
      ("tool_done", str)  —— 某个工具执行完毕（或被人拒绝）
      ("interrupt", str)  —— 需要人工审批，数据是审批问题
      ("end",       None) —— 本轮结束
    """
    stream_input = (
        {"messages": [{"role": "user", "content": user_text}]}
        if user_text is not None
        else Command(resume=resume_value)
    )

    for mode, payload in app.stream(stream_input, config,
                                    stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, meta = payload
            if meta["langgraph_node"] == "llm" and chunk.content:
                yield ("text", chunk.content)

        elif mode == "updates":
            if "__interrupt__" in payload:
                yield ("interrupt", payload["__interrupt__"][0].value)
                continue

            if "llm" in payload:                      # ← 模型这一轮说完话了
                for m in payload["llm"].get("messages", []):
                    for tc in (getattr(m, "tool_calls", None) or []):
                        yield ("tool_call", f"{tc['name']}({tc['args']})")

            elif "tools" in payload:                  # ← tools 节点跑完了
                for m in payload["tools"].get("messages", []):
                    if "人工拒绝" in str(m.content):
                        yield ("tool_done", "⛔ 已拒绝")
                    else:
                        yield ("tool_done", "✅ 工具执行完成")

    yield ("end", None)


