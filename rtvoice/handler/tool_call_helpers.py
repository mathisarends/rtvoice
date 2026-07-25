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
    instructions = [tool.result_instruction] if tool.result_instruction else []
    await send_batched_response(ws, instructions)


async def send_batched_response(
    ws: RealtimeWebSocket, result_instructions: list[str]
) -> None:
    """Send exactly one follow-up response for a completed tool-call batch."""
    if not result_instructions:
        await ws.send(ConversationResponseCreateEvent())
        return

    await ws.send(
        ConversationResponseCreateEvent.from_instructions(
            "\n".join(result_instructions)
        )
    )
