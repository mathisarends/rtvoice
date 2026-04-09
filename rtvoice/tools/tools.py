from __future__ import annotations

import abc
import inspect
import logging
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Self,
    get_args,
    get_origin,
    get_type_hints,
)

from rtvoice.mcp.server import MCPServer
from rtvoice.realtime.schemas import FunctionTool
from rtvoice.tools.registry import (
    RealtimeToolRegistry,
    SubAgentToolRegistry,
    ToolRegistry,
)
from rtvoice.tools.registry.views import Tool
from rtvoice.tools.views import ToolContext, _Inject

if TYPE_CHECKING:
    from rtvoice.tools.registry.views import RealtimeTool, SubAgentTool

logger = logging.getLogger(__name__)

__all__ = ["RealtimeTools", "SubAgentTools", "Tools"]


class BaseTools(abc.ABC):
    """Abstract base class shared by all tool registries.

    Internal helper for shared execution and argument-injection behavior.
    """

    def __init__(self):
        self._registry = self._create_registry()
        self._context: ToolContext = ToolContext()

    @abc.abstractmethod
    def _create_registry(self) -> ToolRegistry:
        """Return the concrete registry implementation for this tool set."""

    def set_context(self, context: ToolContext) -> None:
        self._context = context

    def inject_tool(self, tool: Tool) -> None:
        self._registry.tools[tool.name] = tool

    def eject_tool(self, name: str) -> None:
        self._registry.tools.pop(name, None)

    def register_mcp(self, tool: FunctionTool, server: MCPServer) -> None:
        self._registry.register_mcp(tool, server)

    def get(self, name: str) -> Tool | None:
        return self._registry.get(name)

    def get_tool_schema(self) -> list[FunctionTool]:
        return self._registry.get_tool_schema()

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
        context: ToolContext,
    ) -> dict[str, Any]:
        signature = inspect.signature(tool.function)
        type_hints = get_type_hints(tool.function, include_extras=True)
        arguments = llm_arguments.copy()
        injectable_by_type = self._injectable_by_type_from_context(context)

        for param_name, param in signature.parameters.items():
            if param_name in arguments or param_name in ("self", "cls"):
                continue

            hint = type_hints.get(param_name)
            injected = self._resolve_inject(hint, injectable_by_type)
            if injected is not None:
                arguments[param_name] = injected
            elif param.default is inspect.Parameter.empty:
                raise ValueError(
                    f"Missing required parameter '{param_name}' for tool '{tool.name}'"
                )

        return arguments

    def _injectable_by_type_from_context(self, context: ToolContext) -> dict[type, Any]:
        result: dict[type, Any] = {}
        for field_name in ToolContext.model_fields:
            value = getattr(context, field_name)
            if value is None:
                continue
            result[type(value)] = value
        return result

    def _resolve_inject(
        self, type_hint: Any, injectable_by_type: dict[type, Any]
    ) -> Any | None:
        if type_hint is None or get_origin(type_hint) is not Annotated:
            return None

        args = get_args(type_hint)
        if not any(isinstance(arg, _Inject) for arg in args):
            return None

        requested_type = args[0]
        for _, value in injectable_by_type.items():
            if isinstance(value, requested_type):
                return value

        return None

    def clone(self) -> Self:
        """Create a shallow copy of this tool registry."""
        new = type(self)()
        new._registry.tools = self._registry.tools.copy()
        return new

    def merge(self, other: BaseTools) -> None:
        self._registry.tools.update(other._registry.tools)

    def is_registered(self, tool: Tool) -> bool:
        return tool in self._registry.tools.values()


class Tools(BaseTools):
    """Tool registry for the OpenAI Realtime API.

    This is the user-facing tools class for `RealtimeAgent`.
    """

    def _create_registry(self) -> ToolRegistry:
        return RealtimeToolRegistry()

    def action(
        self,
        description: str,
        name: str | None = None,
        result_instruction: str | None = None,
        holding_instruction: str | None = None,
    ) -> Callable:
        """Register a function as a realtime tool.

        Args:
            description: Natural-language description shown to the model.
            name: Optional tool name override.
            result_instruction: Optional instruction for result phrasing.
            holding_instruction: Optional speech instruction while a delegated
                subagent runs in the background.
        """
        return self._registry.action(
            description,
            name=name,
            result_instruction=result_instruction,
            holding_instruction=holding_instruction,
        )

    def get(self, name: str) -> RealtimeTool | None:
        return self._registry.get(name)

    def get_tool_schema(self) -> list[FunctionTool]:
        return self._registry.get_tool_schema()


class RealtimeTools(Tools):
    """Backward-compatible alias for `Tools`."""


class SubAgentTools(BaseTools):
    """Tool registry for non-realtime (text) agents such as `SubAgent`."""

    def _create_registry(self) -> ToolRegistry:
        return SubAgentToolRegistry()

    def action(
        self,
        description: str,
        name: str | None = None,
        result_instruction: str | None = None,
        status: str | None = None,
        suppress_response: bool = False,
    ) -> Callable:
        return self._registry.action(
            description,
            name=name,
            result_instruction=result_instruction,
            status=status,
            suppress_response=suppress_response,
        )

    def get(self, name: str) -> SubAgentTool | None:
        return self._registry.get(name)

    def get_json_tool_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": tool.model_dump(exclude={"type"}, exclude_none=True),
            }
            for tool in self._registry.get_tool_schema()
        ]
