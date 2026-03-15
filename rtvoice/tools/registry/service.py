import inspect
import re
from collections.abc import Callable

from rtvoice.mcp import MCPServer
from rtvoice.realtime.schemas import FunctionTool
from rtvoice.tools.registry.schema_builder import ToolSchemaBuilder
from rtvoice.tools.registry.views import Tool


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self._schema_builder = ToolSchemaBuilder()

    def action(
        self,
        description: str,
        name: str | None = None,
        result_instruction: str | None = None,
    ):
        def decorator(func: Callable) -> Callable:
            tool = self._build_tool(
                func=func,
                name=name or func.__name__,
                description=description,
                result_instruction=result_instruction,
            )
            self._register_tool(tool)
            return func

        return decorator

    def _validate_status_template(self, status: str, function: Callable) -> None:
        placeholders = {match.group(1) for match in re.finditer(r"\{(\w+)\}", status)}
        param_names = {
            name
            for name in inspect.signature(function).parameters
            if name not in {"self", "cls"}
        }

        unknown_placeholders = placeholders - param_names
        if unknown_placeholders:
            raise ValueError(
                "Status template contains unknown placeholders: "
                f"{unknown_placeholders}. Available parameters: {param_names}"
            )

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def get_tool_schema(self) -> list[FunctionTool]:
        return [tool.to_pydantic() for tool in self.tools.values()]

    def _build_tool(
        self,
        func: Callable,
        name: str,
        description: str,
        result_instruction: str | None,
    ) -> Tool:
        bound_func = getattr(self, func.__name__, func)
        schema = self._schema_builder.build(func)

        return Tool(
            name=name,
            description=description,
            function=bound_func,
            schema=schema,
            result_instruction=result_instruction,
        )

    def _register_tool(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self.tools[tool.name] = tool

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


class RealtimeToolRegistry(ToolRegistry):
    def action(
        self,
        description: str,
        name: str | None = None,
        result_instruction: str | None = None,
        holding_instruction: str | None = None,
    ):
        def decorator(func: Callable) -> Callable:
            from rtvoice.tools.registry.views import RealtimeTool

            bound_func = getattr(self, func.__name__, func)
            schema = self._schema_builder.build(func)
            tool = RealtimeTool(
                name=name or func.__name__,
                description=description,
                function=bound_func,
                schema=schema,
                result_instruction=result_instruction,
                holding_instruction=holding_instruction,
            )
            self._register_tool(tool)
            return func

        return decorator


class SubAgentToolRegistry(ToolRegistry):
    def action(
        self,
        description: str,
        name: str | None = None,
        result_instruction: str | None = None,
        status: str | None = None,
    ):
        def decorator(func: Callable) -> Callable:
            from rtvoice.tools.registry.views import SubAgentTool

            if status is not None:
                self._validate_status_template(status, func)

            bound_func = getattr(self, func.__name__, func)
            schema = self._schema_builder.build(func)
            tool = SubAgentTool(
                name=name or func.__name__,
                description=description,
                function=bound_func,
                schema=schema,
                result_instruction=result_instruction,
                status=status,
            )
            self._register_tool(tool)
            return func

        return decorator
