"""知识库助手 —— HTTP 服务

装配在 lifespan 里做一次（很慢：2 个模型 + MCP 子进程），请求只负责用
app.state.agent。会话状态由 checkpointer 按 thread_id 隔离。

接口一览：
  GET  /              极简前端（演示 SSE 打字机）
  GET  /health        健康检查
  POST /chat          JSON 一问一答
  POST /chat/resume   审批答复，从断点继续
  POST /chat/stream   SSE 流式（推荐）
"""

import json
from contextlib import asynccontextmanager

from fastapi import Depends, Header, HTTPException
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from agent_lg import stream_turn
from bootstrap import build_agent
from config import STATIC_DIR
from ratelimit import limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ★ yield 之前：进程启动时装配一次
    print("正在装配 Agent ...")
    agent, mcp = build_agent()
    app.state.agent = agent
    app.state.mcp = mcp

    yield

    # ★ yield 之后：进程退出时收尾（MCP 子进程必须显式关，否则变孤儿）
    if mcp:
        mcp.stop()


app = FastAPI(title="知识库助手 API", lifespan=lifespan)


# ============ 请求 / 响应模型 ============
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool = True


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    # status: "ok" 正常回答 | "need_approval" 等人工审批 | "paused" 其它暂停
    status: str = "ok"
    question: str | None = None         # status != "ok" 时，要问用户的话


# ============ 公共逻辑 ============
def _make_config(user_id: str, thread_id: str) -> dict:
    """会话级配置。

    ★ C5.2 会话隔离：真正的 thread_id = "{user_id}:{客户端传的值}"

      为什么这样能防越权：
        客户端传什么都被【前缀】上他自己的 user_id。所以 alice 就算填 "bob-session"，
        真实访问到的也是 "alice:bob-session" —— 永远落不到 bob 的命名空间里。

      注意：隔离必须做在【服务端拼装】这一步，绝不能信任客户端传来的完整 id。
    """
    return {
        "recursion_limit": 10,
        "configurable": {
            "thread_id": f"{user_id}:{thread_id}",   # 会话隔离（C5.2）
            "user_id": user_id,                       # ★ 请求级身份，供工具/记忆使用（C5.2b）
        },
    }

def _respond(graph, config, thread_id, result) -> ChatResponse:
    """图跑完了 → 给回复；图停在审批点 → 把问题交回调用方。

    ★ 一次 resume 只解决一个审批：如果模型一口气申请了多个需要审批的工具，
      这里会再返回一次 need_approval，调用方需要再调一次 /chat/resume。
    """
    snap = graph.get_state(config)

    if snap.next:                                   # next 非空 = 图没跑完
        tasks = getattr(snap, "tasks", None) or ()
        interrupts = tasks[0].interrupts if tasks else ()
        if interrupts:
            return ChatResponse(
                reply="",
                thread_id=thread_id,
                status="need_approval",
                question=str(interrupts[0].value),
            )
        # 兜底：图停了但不是等审批（现在不会走到，防的是以后加静态断点）
        return ChatResponse(
            reply="",
            thread_id=thread_id,
            status="paused",
            question=f"图停在 {snap.next}，但不是在等审批",
        )

    return ChatResponse(
        reply=result["messages"][-1].content,
        thread_id=thread_id,
    )


def _sse(kind: str, data) -> str:
    """把一个事件编码成 SSE 报文。

    ★ data 里绝不能出现裸换行 —— 那会破坏 SSE 帧结构，
      所以用 json.dumps 把内容（包括换行）转义成单行文本。
    """
    payload = json.dumps({"type": kind, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"

# （现在先写死在代码里；真实项目放数据库 / Redis）
USERS = {
    "sk-alice-demo-key": "alice",
    "sk-bob-demo-key":   "bob",
}

def current_user(authorization: str = Header(default="")) -> str:
    """从 Authorization 头解析出 user_id。解析不出来 → 401。"""
    scheme, _, key = authorization.partition(" ")
    key = key.strip()

    if scheme.lower() != "bearer" or not key:
        raise HTTPException(
            status_code=401,
            detail="缺少或格式错误的 Authorization 头，应为：Bearer <key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = USERS.get(key)
    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="API Key 无效",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id

def rate_limit(user_id: str = Depends(current_user)) -> str:
    """限流依赖：消费一个令牌，超了就 429。

    ★ 依赖链：rate_limit → current_user → Authorization 头
      所以限流【一定】发生在"已经知道你是谁"之后 ——
      顺序由依赖关系保证，不靠人记得写对顺序。

    ★ 为什么用依赖而不是 middleware：
      middleware 跑在【路由解析之前】，那时还不知道 user_id（认证还没跑）。
      依赖天然插在认证之后，而且可以【按路由】开关（/health 就不用限流）。
    """
    allowed, retry_after = limiter.check(user_id)
    if not allowed:
        # ★ 429 Too Many Requests（RFC 6585）+ Retry-After（RFC 7231）
        #   告诉客户端"等几秒再来"，它才能正确地退避重试。
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，请 {retry_after:.1f} 秒后重试",
            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
        )
    return user_id

# ============ 路由 ============
@app.get("/")
def index():
    """极简前端：输入一句话，看模型逐字吐出来。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user_id: str = Depends(rate_limit)):
    graph = app.state.agent
    config = _make_config(user_id, req.thread_id)

    result = graph.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config,
    )
    return _respond(graph, config, req.thread_id, result)


@app.post("/chat/resume", response_model=ChatResponse)
def chat_resume(req: ResumeRequest, user_id: str = Depends(rate_limit)):
    graph = app.state.agent
    config = _make_config(user_id, req.thread_id)
    result = graph.invoke(
        Command(resume="y" if req.approved else "n"),
        config,
    )
    return _respond(graph, config, req.thread_id, result)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, user_id: str = Depends(rate_limit)):
    """SSE 流式版 /chat：token、工具调用、审批请求都实时推给客户端。"""
    graph = app.state.agent
    config = _make_config(user_id, req.thread_id)

    # stream_turn 是同步生成器 → 用生成器表达式逐条转成 SSE 报文
    body = (_sse(kind, data)
            for kind, data in stream_turn(graph, config, user_text=req.message))

    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",       # 别缓存，否则浏览器攒着一起给
            "X-Accel-Buffering": "no",         # 让 Nginx 之类的反代不要缓冲（云部署时救命）
        },
    )