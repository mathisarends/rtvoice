import json
from typing import Any

from rtvoice.events import EventBus
from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
    FunctionCallItem,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.shared.logging import LoggingMixin
from rtvoice.tools import Tools

_DEFAULT_RESPONSE_INSTRUCTION = (
    "The tool call has completed. Process the result and respond to the user."
)


class ToolCallingWatchdog(LoggingMixin):
    def __init__(self, event_bus: EventBus, tools: Tools, websocket: RealtimeWebSocket):
        self._event_bus = event_bus
        self._tools = tools
        self._websocket = websocket

        self._event_bus.subscribe(FunctionCallItem, self._handle_tool_call)

    async def _handle_tool_call(self, event: FunctionCallItem) -> None:
        tool = self._tools.get(event.name)
        if not tool:
            self.logger.error("Tool '%s' not found", event.name)
            return

        result = await self._tools.execute(event.name, event.arguments or {})

        await self._websocket.send(
            ConversationItemCreateEvent.function_call_output(
                call_id=event.call_id,
                output=self._serialize(result),
            )
        )
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                tool.response_instruction or _DEFAULT_RESPONSE_INSTRUCTION
            )
        )

    def _serialize(self, result: Any) -> str:
        if result is None:
            return "Success"
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
