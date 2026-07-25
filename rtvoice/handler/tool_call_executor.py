from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from transitbus import EventBus

from rtvoice.handler.tool_call_helpers import (
    send_batched_response,
    send_function_call_output,
    serialize_tool_result,
)
from rtvoice.realtime.schemas import FunctionCallItem, ResponseDoneEvent
from rtvoice.realtime.websocket import RealtimeWebSocket

if TYPE_CHECKING:
    from rtvoice.tools import Tools
    from rtvoice.tools.views import Tool

logger = logging.getLogger(__name__)


@dataclass
class _PendingCall:
    call_id: str
    tool: Tool
    task: asyncio.Task


@dataclass
class _ResponseBatch:
    response_id: str
    calls: list[_PendingCall] = field(default_factory=list)


class ToolCallExecutor:
    def __init__(
        self,
        event_bus: EventBus,
        tools: Tools,
        websocket: RealtimeWebSocket,
        supervisor_tool_name: str | None = None,
    ) -> None:
        self._tools = tools
        self._websocket = websocket
        self._supervisor_tool_name = supervisor_tool_name
        self._batches: dict[str, _ResponseBatch] = {}

        event_bus.on(FunctionCallItem, self._on_function_call)
        event_bus.on(ResponseDoneEvent, self._on_response_done)
        logger.debug("ToolCallExecutor initialized")

    async def _on_function_call(self, event: FunctionCallItem) -> None:
        if self._is_supervisor_tool(event.name):
            return

        tool = self._tools.get(event.name)
        if not tool:
            logger.error("Tool '%s' not found", event.name)
            return

        task = asyncio.create_task(
            self._tools.execute(event.name, event.arguments or {})
        )
        batch = self._batches.setdefault(
            event.response_id, _ResponseBatch(response_id=event.response_id)
        )
        batch.calls.append(_PendingCall(call_id=event.call_id, tool=tool, task=task))

    async def _on_response_done(self, event: ResponseDoneEvent) -> None:
        batch = self._batches.pop(event.response_id, None)
        if not batch or not batch.calls:
            return

        results = await asyncio.gather(
            *(call.task for call in batch.calls), return_exceptions=True
        )
        result_instructions: list[str] = []

        for call, result in zip(batch.calls, results, strict=True):
            if isinstance(result, BaseException):
                logger.error("Tool '%s' crashed: %s", call.tool.name, result)
                serialized = f"Tool execution failed: {result}"
            else:
                serialized = serialize_tool_result(result)

            await send_function_call_output(self._websocket, call.call_id, serialized)
            if call.tool.result_instruction:
                result_instructions.append(call.tool.result_instruction)

        await send_batched_response(self._websocket, result_instructions)

    def _is_supervisor_tool(self, tool_name: str | None) -> bool:
        return tool_name == self._supervisor_tool_name
