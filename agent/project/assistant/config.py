"""全项目唯一的配置与路径来源 —— 路径 / 密钥 / 模型 / 提示词

★ 配置来源优先级：环境变量 > 仓库根的 .env
   本地开发靠 .env；容器 / CI / 云平台直接给环境变量，不需要任何文件。
"""

import os
import sys
from pathlib import Path

# ============ 路径（所有"文件在哪"都从这里取）============
BASE_DIR = Path(__file__).resolve().parent

# ---- 只读资源：跟着代码走（烤进镜像）----
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
MCP_SCRIPT = BASE_DIR / "mcp_server.py"
STATIC_DIR = BASE_DIR / "static"

# ---- 可写数据：默认与代码同目录（本地开发），容器里用环境变量指到挂载目录 ----
# ★ 为什么必须可覆盖：
#   SQLite 开 WAL 后会生成 -wal / -shm 两个【兄弟文件】。
#   如果只把 lg_checkpoints.db 单文件挂进容器，兄弟文件就落在容器可写层 ——
#   容器一重建（--build / down）数据全丢。
#   所以：挂【目录】，并让代码知道数据在哪个目录。
DATA_DIR = Path(os.getenv("KB_DATA_DIR", str(BASE_DIR)))

MEMORY_PATH = DATA_DIR / "memory.json"
DB_PATH = DATA_DIR / "lg_checkpoints.db"


# ============ 仓库根：只用于本地开发时找 .env ============
def _find_repo_root() -> Path:
    """向上查找含 common_config.py 的目录（common_config.py 放在仓库根）。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "common_config.py").exists():
            return parent
    return Path(r"H:\ai-learning")  # 兜底：推导失败时保持原有行为


def _load_local_dotenv() -> None:
    """本地开发：把仓库根 .env 里的变量灌进 os.environ。

    ★ 已经有 DEEPSEEK_API_KEY 就直接返回 —— 容器 / CI / 云平台是直接给环境变量的，
      那里根本没有 .env 文件，也不该依赖它。
    """
    if os.getenv("DEEPSEEK_API_KEY"):
        return
    env_file = _find_repo_root() / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)


_load_local_dotenv()


# ============ 模型（环境变量优先，其次 .env 灌进来的值）============
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


# ============ 提示词 ============
SYSTEM_PROMPT = """你是一个个人知识库助手。你有以下能力：
1. 文件操作 — 读取/写入本地文件
2. 知识搜索 — 在本地知识库中搜索文档（关键词 + 语义）
3. 长期记忆 — 记住用户的事实和偏好
4. MCP 工具 — 通过 MCP 协议扩展更多能力

根据用户问题选择合适的工具，用工具结果回答。回答简洁自然。始终用中文回答。"""