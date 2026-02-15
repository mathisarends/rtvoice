import inspect
import json
from collections.abc import Callable
from typing import Any

from rtvoice.events import EventBus
from rtvoice.events.views import ToolCallResultReadyEvent
from rtvoice.realtime.schemas import FunctionCallItem
from rtvoice.shared.logging import LoggingMixin
from rtvoice.tools.registry import ToolRegistry
from rtvoice.tools.registry.views import Tool


class ToolCallingWatchdog(LoggingMixin):
    def __init__(self, event_bus: EventBus, tool_registry: ToolRegistry):
        self._event_bus = event_bus
        self._tool_registry = tool_registry

        self._event_bus.subscribe(
            FunctionCallItem,
            self._handle_tool_call,
        )

    async def _handle_tool_call(self, event: FunctionCallItem) -> None:
        self.logger.info(
            "Tool call received: %s (call_id=%s)",
            event.name,
            event.call_id,
        )

        tool = self._tool_registry.get(event.name)
        if not tool:
            self.logger.error("Tool '%s' not found in registry", event.name)
            return

        try:
            await self._execute_and_dispatch(tool, event)
            self.logger.info(
                "Tool call completed: %s (call_id=%s)",
                event.name,
                event.call_id,
            )
        except Exception as e:
            self.logger.error(
                "Tool execution failed: %s (call_id=%s) - %s",
                event.name,
                event.call_id,
                e,
                exc_info=True,
            )

    async def _execute_and_dispatch(
        self, tool: Tool, call_data: FunctionCallItem
    ) -> None:
        arguments = self._prepare_arguments(tool.function, call_data.arguments or {})
        result = await tool.execute(arguments)
        output = self._serialize_result(result)

        result_event = ToolCallResultReadyEvent(
            call_id=call_data.call_id,
            tool_name=call_data.name,
            output=output,
            response_instruction=tool.response_instruction,
        )
        await self._event_bus.dispatch(result_event)

    def _prepare_arguments(
        self,
        func: Callable[..., Any],
        llm_arguments: dict[str, Any],
    ) -> dict[str, Any]:
        signature = inspect.signature(func)
        arguments = llm_arguments.copy()
        available_special_params = self._get_available_special_params()

        for param_name, param in signature.parameters.items():
            if self._is_managed_parameter(param_name, arguments):
                continue

            if param_name in available_special_params:
                arguments[param_name] = available_special_params[param_name]
            elif param.default == inspect.Parameter.empty:
                raise ValueError(
                    f"Missing required parameter '{param_name}' for tool '{func.__name__}'"
                )

        return arguments

    def _get_available_special_params(self) -> dict[str, Any]:
        return {
            "event_bus": self._event_bus,
        }

    def _is_managed_parameter(self, param_name: str, arguments: dict[str, Any]) -> bool:
        return param_name in arguments or param_name in ("self", "cls")

    def _serialize_result(self, result: Any) -> str:
        if result is None:
            return "Success"

        if isinstance(result, str):
            return result

        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
