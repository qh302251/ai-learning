import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.memory_tools import remember, recall


class TestMemoryTools(unittest.TestCase):
    # def setUp(self):
    #     """覆盖掉之前测试留下的记忆，避免干扰"""
    #     remember("test_key","cleanup")

    @patch("tools.memory_tools.MEMORY_PATH", "test_memory.json")
    def test_remember_and_recall(self):
        """记住一条事实,然后能回忆起来"""
        remember("name", "小明")
        result = recall("name")
        self.assertIn("小明", result)

    @patch("tools.memory_tools.MEMORY_PATH", "test_memory.json")
    def test_recall_not_found(self):
        """搜索不存在的关键词"""
        result = recall("不存在的关键词")
        self.assertIn("找不到", result)

    @patch("tools.memory_tools.MEMORY_PATH", "test_memory.json")
    def test_remember_and_update_existing(self):
        """更新已有key"""
        remember("name", "小明")
        remember("name", "小红")
        result = recall("name")
        self.assertIn("小红", result)

    @patch("tools.memory_tools.MEMORY_PATH", "test_memory.json")
    def test_remember_empty_value(self):
        """储存空字符串值"""
        remember("empty", "")
        result = recall("empty")
        self.assertIn("empty", result)

    @patch("tools.memory_tools.MEMORY_PATH", "test_memory.json")
    def test_recall_with_empty_string(self):
        """传空字符串搜索"""
        remember("name", "小明")
        result = recall("")
        self.assertIn("小明", result)

    @patch("tools.memory_tools.MEMORY_PATH", "test_memory.json")
    def test_remember_emoji(self):
        """记住含表情的文本"""
        remember("mood", "😊")
        result = recall("mood")
        self.assertIn("😊", result)

    @patch("tools.memory_tools.MEMORY_PATH", "test_memory.json")
    def test_remember_chinese_punctuation(self):
        """记住含中文标点的文本"""
        remember("note", "你好！我的世界。")
        result = recall("note")
        self.assertIn("！", result)

    @patch("tools.memory_tools.MEMORY_PATH", "test_memory.json")
    def test_multi_remember_and_recall_partial_match(self):
        """记住多条事实,然后能回忆起来"""
        remember("name", "小明")
        remember("city", "北京")
        remember("age", "23")
        result = recall("name")
        self.assertIn("小明", result)
        result = recall("cit")
        self.assertIn("北京", result)
        result = recall("2")
        self.assertIn("23", result)

    def tearDown(self):
        if os.path.exists("test_memory.json"):
            os.remove("test_memory.json")


if __name__ == "__main__":
    unittest.main()
