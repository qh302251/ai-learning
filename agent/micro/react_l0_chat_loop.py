"""L0 — 纯对话循环（不接任何工具）

本关只有一个目标：搞清楚「多轮对话是怎么记住上下文的」。
这是 ReAct / Agent 的地基 —— 先把这个吃透，L1 再往里面加工具。

跑法：
    & E:\\deeplearning\\envs\\pytorch_env\\python.exe H:\\ai-learning\\agent\\micro\\react_l0_chat_loop.py
"""

import os
import sys

import requests

# 让 common_config 能被 import（文件相对定位，不写死绝对路径，项目可移植）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common_config import API_KEY  # noqa: E402

API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

SYSTEM_PROMPT = "你是一个简洁的助手，回答控制在两句话以内。"


# =====================================================================
# 【参考卡片】调用 API 的"格式" —— 这部分属于查手册，已给全。
# 逐行读注释，不要跳；你要能说清每一行在干嘛。
# =====================================================================
def chat_once(messages):
    """向模型发【一次】请求，返回模型回复的纯文本 (str)。

    messages 是一个 list，每个元素是一个 dict，形如：
        [{"role": "system",    "content": "你是..."},
         {"role": "user",      "content": "你好"},
         {"role": "assistant", "content": "你好呀"}]
    """
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",  # 身份：谁在调用
            "Content-Type": "application/json",  # 我发过去的是 JSON
        },
        json={  # 请求体
            "model": MODEL,
            "messages": messages,  # ★ 把"全部历史"发过去
            # "stream": True,      # 流式是 L3 的事，L0 不用
        },
    )
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# =====================================================================
# 下面的 main() 是【你写】的部分 —— 一共 4 个 TODO。
# =====================================================================
def main():
    messages = []

    # ===== TODO 1：把 system 消息放进去（它必须是第一条）=====
    # 提示：messages.append({"role": ..., "content": ...})
    # SYSTEM_PROMPT 已经在上面写好了，直接用
    messages.append({"role":"system","content":SYSTEM_PROMPT})
    print("（输入 q 退出）")

    while True:
        user_input = input("\n你: ")
        if user_input.strip().lower() == "q":
            break

        # ===== TODO 2：把用户这句话追加进 messages =====
        # 想清楚 role 填什么。
        messages.append({"role":"user","content":user_input})
        # ===== TODO 3：调用 chat_once 拿到回复 =====
        # 一行就够：reply = chat_once(...)
        reply = chat_once(messages)
        # reply = chat_once([messages[0]] + messages[-1:])
        # ===== TODO 4：把模型的回复也追加进 messages ★ 本关最关键的一步 ★
        # 先自己回答两个问题，再用下面的"反证实验"验证：
        #   (a) 这一条 role 填什么？
        #   (b) 为什么必须把它也存进去？
        messages.append({"role":"assistant","content":reply})
        print(f"AI: {reply}")


if __name__ == "__main__":
    main()