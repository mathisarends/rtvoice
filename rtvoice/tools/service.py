from __future__ import annotations

import inspect
import logging
from typing import Any, Self

from rtvoice.mcp.server import MCPServer
from rtvoice.realtime.schemas import FunctionTool
from rtvoice.tools.registry import ToolRegistry
from rtvoice.tools.registry.views import Tool
from rtvoice.tools.views import SpecialToolParameters

logger = logging.getLogger(__name__)


class Tools:
    def __init__(self):
        self._registry = ToolRegistry()
        self._context: SpecialToolParameters = SpecialToolParameters()

    def set_context(self, context: SpecialToolParameters) -> None:
        self._context = context

    def action(
        self,
        description: str,
        name: str | None = None,
        result_instruction: str | None = None,
        is_long_running: bool = False,
        holding_instruction: str | None = None,
    ):
        return self._registry.action(
            description,
            name=name,
            result_instruction=result_instruction,
            is_long_running=is_long_running,
            holding_instruction=holding_instruction,
        )

    def register_mcp(self, tool: FunctionTool, server: MCPServer) -> None:
        self._registry.register_mcp(tool, server)

    def get(self, name: str) -> Tool | None:
        return self._registry.get(name)

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        tool = self._registry.get(name)
        if not tool:
            raise KeyError(f"Tool '{name}' not found in registry")

        prepared = self._prepare_arguments(tool, arguments, self._context)
        return await tool.execute(prepared)

    def _prepare_arguments(
        self,
        tool: Tool,
        llm_arguments: dict[str, Any],
        context: SpecialToolParameters,
    ) -> dict[str, Any]:
        signature = inspect.signature(tool.function)
        arguments = llm_arguments.copy()
        injectable = self._injectable_from_context(context)

        for param_name, param in signature.parameters.items():
            if param_name in arguments or param_name in ("self", "cls"):
                continue

            if param_name in injectable and injectable[param_name] is not None:
                arguments[param_name] = injectable[param_name]
            elif param.default is inspect.Parameter.empty:
                raise ValueError(
                    f"Missing required parameter '{param_name}' for tool '{tool.name}'"
                )

        return arguments

    def _injectable_from_context(
        self, context: SpecialToolParameters
    ) -> dict[str, Any]:
        return {
            field: getattr(context, field)
            for field in SpecialToolParameters.model_fields
        }

    def clone(self) -> Self:
        new = type(self)()
        new._registry.tools = self._registry.tools.copy()
        return new


class RealtimeTools(Tools):
    def get_tool_schema(self) -> list[FunctionTool]:
        return self._registry.get_tool_schema()


class AgentTools(Tools):
    def get_json_tool_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": tool.model_dump(exclude={"type"}, exclude_none=True),
            }
            for tool in self._registry.get_tool_schema()
        ]
