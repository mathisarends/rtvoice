import json
import logging
from typing import Any

from pydantic import BaseModel

from rtvoice.events import EventBus
from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
    FunctionCallItem,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.tools import Tools

_DEFAULT_RESPONSE_INSTRUCTION = (
    "The tool call has completed. Respond directly with the result."
)

logger = logging.getLogger(__name__)


class ToolCallingWatchdog:
    def __init__(self, event_bus: EventBus, tools: Tools, websocket: RealtimeWebSocket):
        self._event_bus = event_bus
        self._tools = tools
        self._websocket = websocket

        self._event_bus.subscribe(FunctionCallItem, self._handle_tool_call)

    async def _handle_tool_call(self, event: FunctionCallItem) -> None:
        tool = self._tools.get(event.name)
        if not tool:
            logger.error("Tool '%s' not found", event.name)
            return

        logger.info(
            "Tool call started: '%s' | args: %s",
            event.name,
            json.dumps(event.arguments or {}, ensure_ascii=False),
        )

        # if tool.pending_message:
        #     await self._websocket.send(
        #         ConversationItemCreateEvent.assistant_message(tool.pending_message)
        #     )

        result = await self._tools.execute(event.name, event.arguments or {})

        logger.info(
            "Tool call result: '%s' | result: %s",
            event.name,
            self._serialize(result),
        )

        await self._websocket.send(
            ConversationItemCreateEvent.function_call_output(
                call_id=event.call_id,
                output=self._serialize(result),
            )
        )
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                tool.result_instruction or _DEFAULT_RESPONSE_INSTRUCTION
            )
        )

    def _serialize(self, result: Any) -> str:
        if result is None:
            return "Success"
        if isinstance(result, str):
            return result
        if isinstance(result, BaseModel):
            return result.model_dump_json(exclude_none=True)
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
