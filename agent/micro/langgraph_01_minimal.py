"""LangGraph 01 — 把 L0 的对话循环画成图（先不接工具）"""

import os
import sys
from typing import Annotated, TypedDict

import requests
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common_config import API_KEY  # noqa: E402

API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"
SYSTEM_PROMPT = "你是一个简洁的助手，回答控制在两句话以内。"


def chat_once(messages):
    """和你 L0 写的一模一样，原样搬过来"""
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "messages": messages},
        timeout=30,
    )
    return resp.json()["choices"][0]["message"]


# =====================================================================
# ① State：声明图里流动什么数据
# =====================================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]      # ← 追加语义，已给


# =====================================================================
# ② 节点：一个函数 = 一个节点
# =====================================================================
ROLE = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}
def llm_node(state):
    # ===== TODO A（就 2 行）=====
    # 1. 从 state 里取出 messages → 调 chat_once
    messages = [
        {"role": ROLE[m.type], "content": m.content}
        for m in state["messages"]
    ]
    reply = chat_once(messages)
    # 2. return {"messages": [回复]}      ← 注意：回复要【包在 list 里】
    return {"messages":[reply]}


# =====================================================================
# ③ 组装图
# =====================================================================
g = StateGraph(State)
g.add_node("llm", llm_node)
g.add_edge(START, "llm")
g.add_edge("llm", END)
app = g.compile()


# =====================================================================
# ④ 跑
# =====================================================================
if __name__ == "__main__":
    print(app.get_graph().draw_mermaid())      # ← ★ 先打印图的结构
    print("=" * 50)
    result = app.invoke({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "你好，我叫小明"},
        ]
    })
    for m in result["messages"]:
        print(" ", m.type, "|", str(m.content)[:40])

    # ===== TODO B（1 行）=====
    # 打印最后一条消息的 content
    print(result["messages"][-1].content)