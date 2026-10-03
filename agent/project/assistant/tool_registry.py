"""工具注册与调度中心"""


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, handler, schema, require_approval=False,
                 needs_context=False):
        """注册一个本地工具。
        ★ needs_context=True：这个工具需要【请求级】信息（如 user_id）。
          这类工具的 handler 要多一个 user_id 参数，但 **绝不能写进 schema** ——
          它是服务端注入的，一旦出现在给模型的工具描述里，
          模型就可能自己编一个 user_id 传进来，隔离就白做了。
        """
        self._tools[name] = {
            "handler": handler,
            "schema": schema,
            "require_approval": require_approval,
            "needs_context": needs_context,
        }

    def needs_approval(self, name):                                      
        return self._tools.get(name, {}).get("require_approval", False)

    def register_mcp_tools(self, schemas, call_fn):
        """注册 MCP 远程工具（通过 call_fn(name, arguments) 调用）"""
        for s in schemas:
            name = s["function"]["name"]
            self._tools[name] = {"handler": call_fn, "schema": s, "is_mcp": True}

    def get_all_schemas(self):
        return [t["schema"] for t in self._tools.values()]

    def dispatch(self, name, context=None, **kwargs):
        """执行工具。
        context：请求级信息，如 {"user_id": "alice"}，由 tools_node 传入。
        ★ 只有声明了 needs_context=True 的工具才会收到 user_id —— 不污染其它工具。
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"未知工具: {name}")

        if tool.get("is_mcp"):
            return tool["handler"](name, kwargs)

        if tool.get("needs_context"):
            user_id = (context or {}).get("user_id")
            # ★ fail-closed：要上下文却拿不到，直接报错。
            #   绝不能"拿不到就当全局"静默放行 —— 那就等于隔离形同虚设。
            if not user_id:
                raise ValueError(f"工具 {name} 需要 user_id，但调用方没有传上下文")
            return tool["handler"](user_id=user_id, **kwargs)

        return tool["handler"](**kwargs)

    def list_tools(self):
        return list(self._tools.keys())
    
