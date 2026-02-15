from collections.abc import Callable

from rtvoice.realtime.schemas import FunctionTool, MCPTool
from rtvoice.tools.registry.views import Tool
from rtvoice.tools.schema import ToolSchemaBuilder


class ToolRegistry:
    def __init__(self, mcp_tools: list[MCPTool] | None = None):
        self.mcp_tools = mcp_tools or []
        self._local_mcp_tools: list[FunctionTool] = []
        self._tools: dict[str, Tool] = {}
        self._schema_builder = ToolSchemaBuilder()

    def action(
        self,
        description: str,
        name: str | None = None,
        response_instruction: str | None = None,
        loading_message: str | None = None,
    ):
        def decorator(func: Callable) -> Callable:
            tool = self._build_tool(
                func=func,
                name=name or func.__name__,
                description=description,
                response_instruction=response_instruction,
                loading_message=loading_message,
            )
            self._register_tool(tool)
            return func

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def add_mcp_tools(self, tools: list[FunctionTool]) -> None:
        self._local_mcp_tools.extend(tools)

    def get_schema(self) -> list[FunctionTool | MCPTool]:
        python_tools = [tool.to_pydantic() for tool in self._tools.values()]
        return python_tools + self._local_mcp_tools + self.mcp_tools

    def _build_tool(
        self,
        func: Callable,
        name: str,
        description: str,
        response_instruction: str | None,
        loading_message: str | None,
    ) -> Tool:
        bound_func = getattr(self, func.__name__, func)
        schema = self._schema_builder.build(func)

        return Tool(
            name=name,
            description=description,
            function=bound_func,
            schema=schema,
            response_instruction=response_instruction,
            loading_message=loading_message,
        )

    def _register_tool(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
