import json
from typing import Any

from pydantic import BaseModel

from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.tools.views import Tool, VoidResult


def serialize_tool_result(result: Any) -> str:
    if isinstance(result, VoidResult):
        return str(result)
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json(exclude_none=True)
    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        return str(result)


async def send_function_call_output(
    ws: RealtimeWebSocket, call_id: str, output: str
) -> None:
    await ws.send(
        ConversationItemCreateEvent.function_call_output(call_id=call_id, output=output)
    )


async def send_response_event(ws: RealtimeWebSocket, tool: Tool) -> None:
    event = (
        ConversationResponseCreateEvent.from_instructions(tool.result_instruction)
        if tool.result_instruction
        else ConversationResponseCreateEvent()
    )
    await ws.send(event)
