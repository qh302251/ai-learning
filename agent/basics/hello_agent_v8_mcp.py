"""
最小 ReAct Agent — 不依赖任何框架，50行实现核心循环
"""

import json
import subprocess
import requests
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from sentence_transformers import SentenceTransformer

# ============ 路径配置 ============
_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "rag")

# ============ 配置 ============
import sys
sys.path.insert(0, r"H:\ai-learning")
from common_config import API_KEY
MODEL = "deepseek-chat"  # 或其他模型
API_URL = "https://api.deepseek.com/v1/chat/completions"  # 改成对应 API 地址


# ============ 工具定义 ============
def read_file(file_path: str) -> str:
    """读取本地文本文件的内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, content: str) -> str:
    """将文本内容写入文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"文件{file_path}写入成功"


def search_documents(keyword: str) -> str:
    """在知识库中搜索包含关键词的文档"""
    try:
        files = os.listdir(os.path.join(_BASE_DIR, "knowledge"))
        txt_files = [f for f in files if f.endswith(".txt")]
        results = []
        for file_name in txt_files:
            file_path = os.path.join(_BASE_DIR, "knowledge", file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if keyword in content:  # 匹配上了，记录结果
                    results.append(f"{file_name} \n {content}")
        if results:
            return "\n---\n".join(results)
        return f"未找到包含 '{keyword}' 的文档"  # 一条都没匹配上时
    except Exception as e:
        return f"搜索知识库时出错: {e}"


def cosine_similarity(a, b):
    """余弦相似度：返回 -1~1,越大越相似"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_search(query: str) -> str:
    """语义搜索，寻找语义最相近的文档"""
    query_vec = _search_model.encode(query)
    best_idx = 0
    best_score = -1
    for i, doc_vec in enumerate(_doc_embeddings):
        score = cosine_similarity(query_vec, doc_vec)
        if score > best_score:
            best_score = score
            best_idx = i
    return f"【{_emb_data['filenames'][best_idx]}】\n{_emb_data['documents'][best_idx]}"


def load_memories():
    if not os.path.exists(os.path.join(_BASE_DIR, "memory.json")):
        return []
    with open(os.path.join(_BASE_DIR, "memory.json"), "r", encoding="utf-8") as f:
        content = f.read().strip()
        return json.loads(content) if content else []


_memory_cache = None


def get_memories():
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = load_memories()
    return _memory_cache


def save_memories():
    with open(os.path.join(_BASE_DIR, "memory.json"), "w", encoding="utf-8") as f:
        json.dump(_memory_cache, f, ensure_ascii=False)


def remember(key: str, value: str) -> str:
    """记住一条事实"""
    memories = get_memories()
    for item in memories:
        if item["key"] == key:
            item["value"] = value
            save_memories()
            return f"已更新 {key}={value}"
    memories.append({"key": key, "value": value})
    save_memories()
    return f"已记住 {key}={value}"


def recall(topic: str) -> str:
    """回忆与topic相关的memory"""
    memory = get_memories()
    matches = []
    keyword = topic.lower()
    for item in memory:
        if keyword in item["key"].lower() or keyword in item["value"].lower():
            matches.append(f"{item['key']}: {item['value']}")
    if not matches:
        return f"找不到与 {topic} 相关的记忆"

    return "找到以下相关的记忆:\n" + "\n".join(matches)


# ============ MCP 客户端 ============
_mcp_proc = None
_mcp_req_id = 0


def mcp_send(method, params=None):
    """发送 JSON-RPC 请求到 MCP Server"""
    global _mcp_req_id
    if _mcp_proc is None:
        raise Exception("MCP Server 未启动")
    _mcp_req_id += 1
    req = {"jsonrpc": "2.0", "id": _mcp_req_id, "method": method}
    if params:
        req["params"] = params
    try:
        _mcp_proc.stdin.write(json.dumps(req) + "\n")
        _mcp_proc.stdin.flush()
        resp = json.loads(_mcp_proc.stdout.readline())
    except Exception as e:
        raise Exception(f"MCP 通信失败: {e}")
    if "error" in resp:
        raise Exception(resp["error"]["message"])
    return resp["result"]


def discover_tools():
    """从 MCP Server 发现工具，转成 OpenAI 兼容的 schema 格式"""
    result = mcp_send("tools/list")
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


def call_mcp_tool(name, arguments):
    """通过 MCP 调用工具"""
    result = mcp_send("tools/call", {"name": name, "arguments": arguments})
    content = result.get("content", [])
    return content[0]["text"] if content else ""


# ============ 语义搜索初始化 ============
print("加载语义搜索模型...")
_search_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
with open(os.path.join(_BASE_DIR, "knowledge", "embeddings.json"), "r", encoding="utf-8") as f:
    _emb_data = json.load(f)
_doc_embeddings = np.array(_emb_data["embeddings"])
print("语义搜索就绪")
# 工具注册表 — dispatch 表，不用 if-elif 长链
TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "search_documents": search_documents,
    "semantic_search": semantic_search,
    "remember": remember,
    "recall": recall,
}

# 工具的 JSON Schema，告诉 LLM 有哪些工具可用
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文本文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写进本地文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件地址"},
                    "content": {"type": "string", "description": "写入内容"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "在本地知识库中搜索包含完整关键词的文本文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "要搜索的关键词"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "寻找与用户语义最接近的文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户的问题原文"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "记住一条关于用户的事实（长期记忆）",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "事实的键，如'研究方向'"},
                    "value": {"type": "string", "description": "事实的值，如'PINN'"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "从长期记忆中查找与某个主题相关的事实",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "要搜索的主题关键词"},
                },
                "required": ["topic"],
            },
        },
    },
]

# ============ Agent 核心循环 ============
# 如果你能正确回答用户的问题，记得在末尾加上嘲讽用户的话
SYSTEM_PROMPT = """
你是一个使用工具的助手。请根据工具返回的结果自然地回答用户。下面是一些示例，请严格遵循它们的回复方式。
记住：你的工具就是你的全部知识，你没有其他的知识来源
用户：请计算sqrt(3)
助手：[调用 calculator 工具]
工具返回：不支持的运算
助手：抱歉，我现在还不会这个运算
"""


def run_agent(messages: list, max_turns: int = 10) -> str:
    """ReAct 循环：处理 tool calls，直到 LLM 给出最终答案（流式输出）"""
    for turn in range(max_turns):
        # 1. 流式调用 LLM
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": messages,
                    "tools": TOOL_SCHEMAS,
                    "stream": True,
                },
                stream=True,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            return "API请求失败，请稍后重试"

        # 逐块累加流式数据
        full_content = ""
        tool_calls_buf = {}  # {index: {id, function: {name, arguments}}}
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"]

            # 内容片段 → 立即打印
            if delta.get("content"):
                full_content += delta["content"]
                print(delta["content"], end="", flush=True)

            # 工具调用片段 → 累加到缓冲区
            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    idx = tc["index"]
                    if idx not in tool_calls_buf:
                        tool_calls_buf[idx] = {
                            "id": "",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.get("id"):
                        tool_calls_buf[idx]["id"] = tc["id"]
                    if "function" in tc:
                        if tc["function"].get("name"):
                            tool_calls_buf[idx]["function"]["name"] += tc["function"][
                                "name"
                            ]
                        if tc["function"].get("arguments"):
                            tool_calls_buf[idx]["function"]["arguments"] += tc[
                                "function"
                            ]["arguments"]

        print()  # 流式结束，换行

        # 2. 检查是否有工具调用
        if tool_calls_buf:
            # 构造 assistant 消息（包含 tool_calls）
            tool_calls_list = [
                {
                    "id": tool_calls_buf[idx]["id"],
                    "type": "function",
                    "function": {
                        "name": tool_calls_buf[idx]["function"]["name"],
                        "arguments": tool_calls_buf[idx]["function"]["arguments"],
                    },
                }
                for idx in sorted(tool_calls_buf.keys())
            ]
            assistant_msg = {"role": "assistant", "content": full_content or None}
            assistant_msg["tool_calls"] = tool_calls_list
            messages.append(assistant_msg)

            # 执行每个工具
            for tc_info in tool_calls_list:
                func_name = tc_info["function"]["name"]
                try:
                    args = json.loads(tc_info["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                print(f"  → 调用 {func_name}({args})")

                if func_name in TOOLS:
                    # 本地工具
                    try:
                        result = TOOLS[func_name](**args)
                    except Exception as e:
                        result = f"工具执行出错: {e}"
                else:
                    # MCP 远程工具
                    try:
                        result = call_mcp_tool(func_name, args)
                    except Exception as e:
                        result = f"MCP 调用失败: {e}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_info["id"],
                        "content": str(result),
                    }
                )
            continue

        # 没有工具调用 → 最终答案
        messages.append({"role": "assistant", "content": full_content})
        return full_content

    return "超过最大轮数"


# ============ 测试 ============
if __name__ == "__main__":
    # 1. 启动 MCP Server
    print("启动 MCP Server...")
    try:
        _mcp_proc = subprocess.Popen(
            ["python", os.path.join(_BASE_DIR, "..", "mcp", "mcp_server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        mcp_schemas = discover_tools()
        TOOL_SCHEMAS[:] = mcp_schemas + TOOL_SCHEMAS  # MCP 放前面，本地放后面
        print(f"🔧 MCP 就绪，发现 {len(mcp_schemas)} 个远程工具")
    except Exception as e:
        print(f"⚠️ MCP Server 启动失败: {e}")
        print("将仅使用本地工具")

    # 2. 加载记忆
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    memories = get_memories()
    if memories:
        memory_text = "已知信息：" + ";".join(
            [f"{m['key']}={m['value']}" for m in memories]
        )
        messages[0]["content"] += "\n" + memory_text
        print(f"📝 已加载 {len(memories)} 条记忆")

    print("🤖 Agent 已启动！输入 exit 退出\n")

    while True:
        user_input = input(">>> ")
        if user_input.lower() == "exit":
            if _mcp_proc:
                _mcp_proc.stdin.close()
                _mcp_proc.wait()
            print("再见！")
            break

        messages.append({"role": "user", "content": user_input})
        run_agent(messages)
        print()  # 轮次之间空一行
