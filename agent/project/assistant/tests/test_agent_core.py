import unittest
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_core import run_agent
from tool_registry import ToolRegistry

class FakeStreamingResponse:
    """模拟 API 的流式响应"""
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for line in self._lines:
            yield line

class TestRunAgent(unittest.TestCase):

    @patch("agent_core.requests.post")
    def test_no_tool_call(self, mock_post):
        """API 直接返回文字，没有工具调用"""
        messages = [{"role": "user", "content": "你好"}]
        registry = ToolRegistry()
        api_config = {
            "api_key": "test-key",
            "api_url": "https://fake.url",
            "model": "test-model",
        }

        # 模拟流式返回
        fake_lines = [
            b'data: {"choices":[{"delta":{"content":"hello!"}}]}',
            b'data: [DONE]',
        ]
        mock_post.return_value = FakeStreamingResponse(fake_lines)

        result = run_agent(messages, registry, api_config)
        self.assertEqual(result, "hello!")


    @patch("agent_core.requests.post")
    def test_tool_call_then_answer(self, mock_post):
        """agent 调用工具后，再返回最终答案"""
        messages = [{"role": "user", "content": "1+2等于多少？"}]
        registry = ToolRegistry()
        registry.register("add", lambda x, y: x + y, {"name": "add"})
        api_config = {"api_key": "test-key", "api_url": "https://fake.url", "model": "test-model"}

        # 第一次响应：要求调用 add 工具
        # 第二次响应：最终答案
        add_call = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call1","function":{"name":"add","arguments":"{\\"x\\":1,\\"y\\":2}"}}]}}]}',
            b'data: [DONE]',
            ]
        final_answer = [
            b'data: {"choices":[{"delta":{"content":"1+2=3"}}]}',
            b'data: [DONE]',
        ]
        mock_post.side_effect = [
            FakeStreamingResponse(add_call),
            FakeStreamingResponse(final_answer),
        ]

        result = run_agent(messages, registry, api_config)
        self.assertEqual(result, "1+2=3")  

if __name__ == "__main__":
    unittest.main()          