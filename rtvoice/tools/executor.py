from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.state.base import VoiceAssistantEvent
from rtvoice.tools.models import (
    FunctionCallItem,
    FunctionCallResult,
    SpecialToolParameters,
)
from rtvoice.tools.registry import Tool, ToolRegistry

if TYPE_CHECKING:
    from rtvoice.events.bus import EventBus
    from rtvoice.realtime.messaging.message_manager import RealtimeMessageManager


class ToolExecutor(LoggingMixin):
    def __init__(
        self,
        tool_registry: ToolRegistry,
        message_manager: RealtimeMessageManager,
        special_tool_parameters: SpecialToolParameters,
        event_bus: EventBus,
    ):
        self._tool_registry = tool_registry
        self._message_manager = message_manager
        self._special_tool_parameters = special_tool_parameters
        self._event_bus = event_bus

        self._event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_STARTED_TOOL_CALL, self._handle_tool_call
        )

    async def _handle_tool_call(
        self, event: VoiceAssistantEvent, data: FunctionCallItem
    ) -> None:
        function_name = data.name
        llm_arguments = data.arguments or {}

        tool = self._tool_registry.get(function_name)

        if tool.loading_message:
            await self._message_manager.send_loading_message(tool.loading_message)

        await self._execute_tool(tool, data, llm_arguments)

    async def _execute_tool(
        self, tool: Tool, data: FunctionCallItem, llm_arguments: dict[str, Any]
    ) -> None:
        final_arguments = self._build_final_arguments(tool.function, llm_arguments)
        result = await tool.execute(final_arguments)
        await self._send_result_and_notify(data, result, tool.response_instruction)

    async def _send_result_and_notify(
        self,
        call_data: FunctionCallItem,
        result: Any,
        response_instruction: str | None = None,
    ) -> None:
        function_call_result = FunctionCallResult(
            tool_name=call_data.name,
            call_id=call_data.call_id,
            output=result,
            response_instruction=response_instruction,
        )
        await self._message_manager.send_tool_result(function_call_result)
        await self._notify_tool_finished()

    async def _notify_tool_finished(self) -> None:
        await self._event_bus.publish_async(
            VoiceAssistantEvent.ASSISTANT_RECEIVED_TOOL_CALL_RESULT
        )

    def _build_final_arguments(
        self, func: callable, llm_arguments: dict[str, Any]
    ) -> dict[str, Any]:
        signature = inspect.signature(func)
        final_arguments = llm_arguments.copy()
        special_param_names = set(SpecialToolParameters.model_fields.keys())

        for param_name, param in signature.parameters.items():
            if self._should_skip_parameter(param_name, final_arguments):
                continue

            if param_name in special_param_names:
                self._inject_special_parameter_if_available(
                    param_name, param, final_arguments
                )

        return final_arguments

    def _should_skip_parameter(
        self, param_name: str, current_arguments: dict[str, Any]
    ) -> bool:
        return param_name in current_arguments or param_name in ("self", "cls")

    def _inject_special_parameter_if_available(
        self,
        param_name: str,
        param: inspect.Parameter,
        arguments: dict[str, Any],
    ) -> None:
        injected_value = getattr(self._special_tool_parameters, param_name, None)

        if injected_value is not None:
            arguments[param_name] = injected_value
        elif param.default == inspect.Parameter.empty:
            raise ValueError(
                f"Required special parameter '{param_name}' is not available"
            )
