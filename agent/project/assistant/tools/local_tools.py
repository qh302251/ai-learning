"""本地文件操作工具"""

import os


def read_file(file_path: str) -> str:
    """读取本地文本文件的内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, content: str) -> str:
    """将文本内容写入文件（自动创建父目录）"""
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"文件 {file_path} 写入成功"


def delete_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"文件不存在:{file_path}"
    os.remove(file_path)
    return f"文件已删除:{file_path}"


def register_all(registry):
    registry.register(
        "read_file",
        read_file,
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取本地文本文件的内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "文件路径"},
                    },
                    "required": ["file_path"],
                },
            },
        },
    )
    registry.register(
        "write_file",
        write_file,
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "将文本内容写入本地文件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "文件路径"},
                        "content": {"type": "string", "description": "写入内容"},
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        require_approval=True,
    )
    registry.register(
        "delete_file",
        delete_file,
        {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "删除本地文件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "文件路径"},
                    },
                    "required": ["file_path"],
                },
            },
        },
        require_approval=True,
    )
