import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tool_registry import ToolRegistry

class TestToolRegistry(unittest.TestCase):

    def test_register_and_list(self):
        """注册一个工具后，list_tool能返回它的名字"""
        registry=ToolRegistry()
        registry.register("hello", lambda:"hi", {"name": "hello"})
        tools=registry.list_tools()
        self.assertIn("hello",tools) 
    
    def test_get_all_shcemas(self):
        """注册工具后，get_all_shcemas返回正确的shcemas"""
        registry=ToolRegistry()
        schema={"name":"greet","description":"打招呼"}
        registry.register("greet",lambda name:f"name:你好{name}",schema)
        schemas=registry.get_all_schemas()
        self.assertIn(schema,schemas)
        self.assertEqual(len(schemas),1)

    def test_dispatch_calls_handler(self):
        """dispatch能正确调用注册的工具"""
        registry=ToolRegistry()
        registry.register("add",lambda x,y: x+y, {"name":"add"})
        result=registry.dispatch("add",x=1,y=2)
        self.assertEqual(result,3)

    def test_dispatch_unknown_tool(self):
        """调用未注册的dispatch应该报ValueError"""
        registry=ToolRegistry()
        with self.assertRaises(ValueError):
            registry.dispatch("未注册的工具")

    def test_register_mcp_tool(self):
        """注册MCP工具后，dispatch会通过call_fn调用"""
        registry=ToolRegistry()
        schemas=[{"function":{"name":"mcp_tool1"}}]
        call_log=[]
        def fake_call(name,args):
            call_log.append((name,args))
            return f"{name} 结果"
        registry.register_mcp_tools(schemas, fake_call)
        self.assertIn("mcp_tool1", registry.list_tools())

        result = registry.dispatch("mcp_tool1", arg1=1)
        self.assertEqual(result, "mcp_tool1 结果")
        self.assertEqual(call_log, [("mcp_tool1", {"arg1": 1})])

if __name__ == "__main__":
    unittest.main()
