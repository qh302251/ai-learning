"""LangGraph 01 — 把 L0 的对话循环画成图（先不接工具）"""

import os
import sys
from typing import Annotated, TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common_config import API_KEY  # noqa: E402

SYSTEM_PROMPT = "你是一个简洁的助手，回答控制在两句话以内。"

def get_weather(city):
    return f"{city}的温度是23摄氏度，天气晴"

def calculator(operation,a,b):
    if operation == "add":
        return a+b
    elif operation == "multiply":
        return a*b
TOOL_FUNCS={
    "get_weather":get_weather,
    "calculator":calculator,
}
TOOL_SCHEMAS = [
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
            "name": "calculator",
            "description": "加法和乘法",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                    "type": "string",
                    "enum": ["add", "multiply"],   
                    },
                    "a": {"type": "number"},      
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        },
    }
]

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1",
)
llm_tools = llm.bind_tools(TOOL_SCHEMAS)
# =====================================================================
# ① State：声明图里流动什么数据
# =====================================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]      

# =====================================================================
# ② 节点：一个函数 = 一个节点
# =====================================================================
def llm_node(state):
    return {"messages": [llm_tools.invoke(state["messages"])]}

NEED_APPROVAL = {"calculator"}

def tools_node(state):
    last = state["messages"][-1]
    out = []
    for tc in last.tool_calls:
        if tc["name"] in NEED_APPROVAL:
            answer = interrupt(f"要执行 {tc['name']}({tc['args']})，批准吗？(y/n)")   # ★ 停在这
            if answer != "y":
                out.append(ToolMessage(
                    content="【人工拒绝】用户未批准该操作。",
                    tool_call_id=tc["id"],
                ))
                continue
        result = TOOL_FUNCS[tc["name"]](**tc["args"])
        out.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": out}

def should_continue(state):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"          # 有申请 → 去 tools 节点
    return END                  # 没申请 → 结束
# =====================================================================
# ③ 组装图
# =====================================================================
g = StateGraph(State)
g.add_node("llm", llm_node)
g.add_edge(START, "llm")


g.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
g.add_node("tools", tools_node)
g.add_edge("tools", "llm")

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
app = g.compile(
    checkpointer=SqliteSaver(conn),
    )


# =====================================================================
# ④ 跑
# =====================================================================
if __name__ == "__main__":
    print("=" * 50)
    config = {"recursion_limit": 10, "configurable": {"thread_id": "user-1"}}

    state = app.invoke({"messages": [{"role": "user", "content": "3乘7等于多少"}]}, config)

    if state.get("__interrupt__"):
        print("★ 停下等审批:", state["__interrupt__"][0].value)
        state = app.invoke(Command(resume=input("批准吗？(y/n) ")), config)
    else:
        print("★ 没停，直接跑完")

    print("AI:", state["messages"][-1].content)
    



