from __future__ import annotations

import inspect
import logging
import re
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

from pydantic import BaseModel

if TYPE_CHECKING:
    from rtvoice.mcp.server import MCPServer

from rtvoice.realtime.schemas import FunctionTool
from rtvoice.tools.di import ToolContext, _Inject
from rtvoice.tools.schema_builder import ToolSchemaBuilder
from rtvoice.tools.views import Tool

logger = logging.getLogger(__name__)


class Tools:
    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self._schema_builder = ToolSchemaBuilder()
        self._context: ToolContext | None = None

    def action(
        self,
        description: str,
        name: str | None = None,
        param_model: type[BaseModel] | None = None,
        result_instruction: str | None = None,
        holding_instruction: str | None = None,
        status: str | Callable | None = None,
        steering: str | None = None,
    ) -> Callable:
        def decorator(func: Callable) -> Callable:
            if isinstance(status, str):
                self._validate_status_template(status, func, param_model)

            bound_func = getattr(self, func.__name__, func)
            schema = self._schema_builder.build(func, param_model=param_model)
            tool = Tool(
                name=name or func.__name__,
                description=description,
                function=bound_func,
                schema=schema,
                param_model=param_model,
                result_instruction=result_instruction,
                holding_instruction=holding_instruction,
                status=status,
                steering=steering,
            )
            self._register_tool(tool)
            return func

        return decorator

    def set_context(self, context: ToolContext) -> None:
        self._context = context

    def inject_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def eject_tool(self, name: str) -> None:
        self.tools.pop(name, None)

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

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def get_steering(self, name: str) -> str | None:
        tool = self.tools.get(name)
        if tool is None:
            return None
        return tool.steering

    def get_tool_schema(self) -> list[FunctionTool]:
        return [tool.to_pydantic() for tool in self.tools.values()]

    def get_json_tool_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": tool.model_dump(exclude={"type"}, exclude_none=True),
            }
            for tool in self.get_tool_schema()
        ]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        tool = self.get(name)
        if not tool:
            raise KeyError(f"Tool '{name}' not found in registry")

        prepared = self._prepare_arguments(tool, arguments, self._context)
        return await tool.execute(prepared)

    def clone(self) -> Self:
        new = type(self)()
        new.tools = self.tools.copy()
        return new

    def merge(self, other: Tools) -> None:
        self.tools.update(other.tools)

    def is_registered(self, tool: Tool) -> bool:
        return tool in self.tools.values()

    def _register_tool(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self.tools[tool.name] = tool

    def _validate_status_template(
        self, status: str, function: Callable, param_model: type[BaseModel] | None
    ) -> None:
        placeholders = {match.group(1) for match in re.finditer(r"\{(\w+)\}", status)}
        if not placeholders:
            return

        if param_model is not None:
            available_names = set(param_model.model_fields.keys())
        else:
            available_names = {
                name
                for name in inspect.signature(function).parameters
                if name not in {"self", "cls"}
            }

        unknown_placeholders = placeholders - available_names
        if unknown_placeholders:
            raise ValueError(
                "Status template contains unknown placeholders: "
                f"{unknown_placeholders}. Available parameters: {available_names}"
            )

    def _prepare_arguments(
        self,
        tool: Tool,
        llm_arguments: dict[str, Any],
        context: ToolContext | None,
    ) -> dict[str, Any]:
        signature = inspect.signature(tool.function)
        type_hints = get_type_hints(tool.function, include_extras=True)
        injectable_by_type = self._injectable_by_type_from_context(context)

        if tool.param_model is not None:
            return self._prepare_with_param_model(
                tool, llm_arguments, signature, type_hints, injectable_by_type
            )

        arguments = llm_arguments.copy()
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

    def _prepare_with_param_model(
        self,
        tool: Tool,
        llm_arguments: dict[str, Any],
        signature: inspect.Signature,
        type_hints: dict[str, Any],
        injectable_by_type: dict[type, Any],
    ) -> dict[str, Any]:
        model_instance = tool.param_model(**llm_arguments)
        arguments: dict[str, Any] = {}

        for param_name, param in signature.parameters.items():
            if param_name in ("self", "cls"):
                continue

            hint = type_hints.get(param_name)
            unwrapped = get_args(hint)[0] if get_origin(hint) is Annotated else hint

            if unwrapped is tool.param_model:
                arguments[param_name] = model_instance
                continue

            injected = self._resolve_inject(hint, injectable_by_type)
            if injected is not None:
                arguments[param_name] = injected
            elif param.default is inspect.Parameter.empty:
                raise ValueError(
                    f"Missing required parameter '{param_name}' for tool '{tool.name}'"
                )

        return arguments

    def _injectable_by_type_from_context(
        self, context: ToolContext | None
    ) -> dict[type, Any]:
        result: dict[type, Any] = {}
        if context is None:
            return result

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
