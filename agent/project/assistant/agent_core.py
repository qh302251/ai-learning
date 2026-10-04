"""ReAct Agent 核心循环 — 与工具解耦，接收 ToolRegistry 即可运行"""

import json

import requests


def run_agent(
    messages: list,
    registry,
    api_config: dict,
    max_turns: int = 10,
    user_id: str | None = None,
) -> str:
    """ReAct 主循环

    Args:
        messages: 对话消息列表
        registry: ToolRegistry 实例
        api_config: 字典，包含 api_key, api_url, model
        max_turns: 最大工具调用轮数
        user_id: 请求级用户标识，透传给【需要上下文】的工具（记忆类）。
                 ★ 默认 None 是故意的（fail-closed）：
                   需要 user_id 的工具会明确报错，而不是"拿不到就当全局"静默放行 ——
                   那等于多用户隔离形同虚设。
    Returns:
        最终回复文本
    """
    api_key = api_config["api_key"]
    api_url = api_config["api_url"]
    model = api_config.get("model", "deepseek-chat")
    tool_schemas = registry.get_all_schemas()

    for turn in range(max_turns):
        # 1. 调用 LLM（流式）
        try:
            response = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "tools": tool_schemas,
                    "stream": True,
                },
                stream=True,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"\nAPI 请求失败: {e}")
            return "抱歉，API 请求失败，请稍后重试"

        # 2. 流式读取
        full_content = ""
        tool_calls_buf = {}

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

            if delta.get("content"):
                full_content += delta["content"]
                print(delta["content"], end="", flush=True)

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

        print()

        # 3. 有工具调用 → 执行工具
        if tool_calls_buf:
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
            messages.append(
                {
                    "role": "assistant",
                    "content": full_content or None,
                    "tool_calls": tool_calls_list,
                }
            )

            for tc in tool_calls_list:
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                print(f"  -> 调用 {func_name}({args})")

                try:
                    # ★ 请求级上下文，与 agent_lg.py 的 tools_node 保持一致。
                    #   registry 只把它转交给声明了 needs_context=True 的工具，
                    #   其它工具完全感知不到 —— 不污染它们的签名。
                    result = registry.dispatch(
                        func_name,
                        context={"user_id": user_id},
                        **args,
                    )
                except Exception as e:
                    result = f"工具执行出错: {e}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    }
                )
            continue

        # 4. 无工具调用 → 最终答案
        messages.append({"role": "assistant", "content": full_content})
        return full_content

    return "超过最大轮数，对话已结束"
