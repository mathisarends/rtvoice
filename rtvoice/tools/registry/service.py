from collections.abc import Callable

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

    def get_schema(self) -> list[FunctionTool]:
        return [tool.to_pydantic() for tool in self._tools.values()]

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
