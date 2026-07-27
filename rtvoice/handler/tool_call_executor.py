from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from transitbus import EventBus

from rtvoice.events.views import (
    ToolExecutedEvent,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
)
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
    ) -> None:
        self._event_bus = event_bus
        self._tools = tools
        self._websocket = websocket
        self._batches: dict[str, _ResponseBatch] = {}

        self._event_bus.on(FunctionCallItem, self._on_function_call)
        self._event_bus.on(ResponseDoneEvent, self._on_response_done)
        logger.debug("ToolCallExecutor initialized")

    async def _on_function_call(self, event: FunctionCallItem) -> None:
        tool = self._tools.get(event.name)
        if not tool:
            logger.error("Tool '%s' not found", event.name)
            return

        batch = self._batches.get(event.response_id)
        if batch is None:
            batch = _ResponseBatch(response_id=event.response_id)
            self._batches[event.response_id] = batch
            await self._event_bus.dispatch(
                ToolExecutionStartedEvent(response_id=event.response_id)
            )

        task = asyncio.create_task(
            self._tools.execute(event.name, event.arguments or {})
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
        should_respond = False

        for call, result in zip(batch.calls, results, strict=True):
            if isinstance(result, BaseException):
                logger.error("Tool '%s' crashed: %s", call.tool.name, result)
                serialized = f"Tool execution failed: {result}"
                call_should_respond = True
            else:
                serialized = serialize_tool_result(result)
                if result.respond is not None:
                    call_should_respond = result.respond
                else:
                    call_should_respond = call.tool.respond if result.ok else True

            should_respond |= call_should_respond
            await self._event_bus.dispatch(
                ToolExecutedEvent(
                    name=call.tool.name,
                    action_kind=call.tool.kind,
                    silent=not call_should_respond,
                )
            )

            await send_function_call_output(self._websocket, call.call_id, serialized)
            if call.tool.result_instruction:
                result_instructions.append(call.tool.result_instruction)

        if should_respond:
            await send_batched_response(self._websocket, result_instructions)
        await self._event_bus.dispatch(
            ToolExecutionCompletedEvent(
                response_id=batch.response_id,
                response_pending=should_respond,
            )
        )
