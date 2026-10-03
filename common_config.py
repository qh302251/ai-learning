# -*- coding: utf-8 -*-
"""统一读取本地密钥 —— 所有脚本都从这里拿 API Key，不再硬编码。

密钥存放在 H:/ai-learning/.env（已被 .gitignore 忽略，绝不进 git）。
换 key 时只改 .env 一行即可，不用翻 17 个文件。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
MODEL   = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def require_key() -> str:
    """取 key；没配就抛出清晰报错，而不是让人对着 401 发呆。"""
    if not API_KEY:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY。请确认 %s 存在且包含该变量（可从 .env.example 复制）。" % _ENV_PATH
        )
    return API_KEY