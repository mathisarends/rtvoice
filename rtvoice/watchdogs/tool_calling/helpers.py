import json
from typing import Any

from pydantic import BaseModel

from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.subagent.views import SubAgentResult
from rtvoice.tools.views import Tool, VoidResult


class ToolCallWebSocketHelper:
    def __init__(self, websocket: RealtimeWebSocket) -> None:
        self._websocket = websocket

    def serialize(self, result: Any) -> str:
        if isinstance(result, VoidResult):
            return str(result)
        if isinstance(result, SubAgentResult):
            return result.to_agent_output()
        if isinstance(result, str):
            return result
        if isinstance(result, BaseModel):
            return result.model_dump_json(exclude_none=True)
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)

    async def send_function_call_output(self, call_id: str, output: str) -> None:
        await self._websocket.send(
            ConversationItemCreateEvent.function_call_output(
                call_id=call_id, output=output
            )
        )

    async def send_response_event(self, tool: Tool) -> None:
        if tool.result_instruction:
            event = ConversationResponseCreateEvent.from_instructions(
                tool.result_instruction
            )
        else:
            event = ConversationResponseCreateEvent()
        await self._websocket.send(event)
