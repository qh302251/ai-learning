"""长期记忆工具 — 持久化存储用户事实

★ C5.2b：记忆按 user_id 隔离。
  每条记录都带 user_id，读写都必须【同时】按 user_id 过滤 ——
  以前是"全局一个数组"，任何用户都读得到别人的，那就是跨用户泄露。
"""

import json
import os

from config import MEMORY_PATH   # ★ 路径只有 config 一个来源

# 直连工具（单元测试 / CLI）时的默认归属
DEFAULT_USER = "default"

# 老数据的归属：加 user_id 之前写下的记录【没有主人信息】，
# 所以只能人工指定一个。★ 生产环境这是个判断题：
#   要么按业务指定默认主人，要么干脆隔离掉不用（宁可丢，不可泄露）。
LEGACY_USER = "alice"

_cache = None


def _migrate(records: list) -> None:
    """惰性迁移：给没有 user_id 的老记录补上归属。只在第一次读文件时跑一次。"""
    changed = False
    for r in records:
        if "user_id" not in r:
            r["user_id"] = LEGACY_USER
            changed = True
    if changed:
        _save()


def _load() -> list:
    global _cache
    if _cache is not None:
        return _cache
    if not os.path.exists(MEMORY_PATH):
        _cache = []
        return _cache
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        _cache = json.loads(content) if content else []
    _migrate(_cache)
    return _cache


def _save() -> None:
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False)


def get_all(user_id: str = DEFAULT_USER) -> list:
    """取【某个用户】的全部记忆。★ 注意：不再是"全部记忆"。"""
    return [m for m in _load() if m.get("user_id") == user_id]


def remember(key: str, value: str, user_id: str = DEFAULT_USER) -> str:
    """记住一条关于用户的事实（只影响这个 user_id 自己的记忆）"""
    memories = _load()
    for item in memories:
        # ★ 必须【同时】匹配 user_id 和 key。
        #   只匹配 key 的话，bob 的"名字"会覆盖掉 alice 的"名字"。
        if item.get("user_id") == user_id and item["key"] == key:
            item["value"] = value
            _save()
            return f"已更新 {key}={value}"
    memories.append({"user_id": user_id, "key": key, "value": value})
    _save()
    return f"已记住 {key}={value}"


def recall(topic: str, user_id: str = DEFAULT_USER) -> str:
    """从【该用户自己的】长期记忆中查找与 topic 相关的事实"""
    keyword = topic.lower()
    matches = [
        f"{m['key']}: {m['value']}"
        for m in _load()
        if m.get("user_id") == user_id
        and (keyword in m["key"].lower() or keyword in m["value"].lower())
    ]
    if not matches:
        return f"找不到与 '{topic}' 相关的记忆"
    return "找到以下相关的记忆:\n" + "\n".join(matches)


def render_for(user_id: str) -> str:
    """把某个用户的记忆渲染成一段文本，供 system prompt 使用。

    ★ 这是"记忆进提示词"的【唯一入口】。
      以前用的是 get_all()（全量、无参数），那正是跨用户泄露的源头。
    """
    memories = get_all(user_id)
    if not memories:
        return ""
    return "已知信息：" + "; ".join(f"{m['key']}={m['value']}" for m in memories)

def stats() -> dict:
    """运维用：每个 user 名下有几条记忆 —— 只返回【数量】，不返回内容。

    ★ 这是唯一允许"跨用户"看的接口，但它不泄露任何内容，所以是安全的。
      名字刻意不叫 get_all，就是为了不让人误用它去取数据。
    """
    out = {}
    for m in _load():
        uid = m.get("user_id")
        out[uid] = out.get(uid, 0) + 1
    return out

def register_all(registry):
    registry.register(
        "remember",
        remember,
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": "记住一条关于用户的事实（长期记忆），如偏好、个人信息等",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "事实的键，如'研究方向'",
                        },
                        "value": {
                            "type": "string",
                            "description": "事实的值，如'PINN'",
                        },
                    },
                    "required": ["key", "value"],
                },
            },
        },
        needs_context=True,          # ★ 需要 user_id（但 user_id 不在 schema 里！）
    )
    registry.register(
        "recall",
        recall,
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "从长期记忆中查找与某个主题相关的事实",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "要搜索的主题关键词",
                        },
                    },
                    "required": ["topic"],
                },
            },
        },
        needs_context=True,          # ★ 同上
    )