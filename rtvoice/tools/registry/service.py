import inspect
from collections.abc import Callable

from rtvoice.mcp import MCPServer
from rtvoice.realtime.schemas import FunctionTool
from rtvoice.tools.registry.schema_builder import ToolSchemaBuilder
from rtvoice.tools.registry.views import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._schema_builder = ToolSchemaBuilder()

    def action(
        self,
        description: str,
        name: str | None = None,
        result_instruction: str | None = None,
        suppress_response: bool = False,
        is_subagent: bool = False,
    ):
        def decorator(func: Callable) -> Callable:
            tool = self._build_tool(
                func=func,
                name=name or func.__name__,
                description=description,
                result_instruction=result_instruction,
                suppress_response=suppress_response,
                is_subagent=is_subagent,
            )
            self._register_tool(tool)
            return func

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_tool_schema(self) -> list[FunctionTool]:
        return [tool.to_pydantic() for tool in self._tools.values()]

    def _build_tool(
        self,
        func: Callable,
        name: str,
        description: str,
        result_instruction: str | None,
        suppress_response: bool = False,
        is_subagent: bool = False,
    ) -> Tool:
        bound_func = getattr(self, func.__name__, func)
        schema = self._schema_builder.build(func)

        return Tool(
            name=name,
            description=description,
            function=bound_func,
            schema=schema,
            result_instruction=result_instruction,
            suppress_response=suppress_response,
            is_subagent=is_subagent,
        )

    def _register_tool(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def register_mcp(self, tool: FunctionTool, server: MCPServer) -> None:
        async def handler(**kwargs):
            return await server.call_tool(tool.name, kwargs or None)

        handler.__name__ = tool.name
        handler.__signature__ = inspect.Signature()

        mcp_tool = Tool(
            name=tool.name,
            description=tool.description or "",
            function=handler,
            schema=tool.parameters,
        )
        self._register_tool(mcp_tool)
