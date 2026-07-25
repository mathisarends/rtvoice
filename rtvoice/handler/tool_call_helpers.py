from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel

from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
)
from rtvoice.realtime.websocket import RealtimeWebSocket

if TYPE_CHECKING:
    from rtvoice.tools.results import ActionResult
    from rtvoice.tools.views import Tool


def serialize_tool_result(result: ActionResult) -> str:
    if not result.ok:
        return result.error or "Tool execution failed."

    value = result.value
    if value is None:
        return "OK"
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json(exclude_none=True)
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


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
    if not result_instructions:
        await ws.send(ConversationResponseCreateEvent())
        return

    await ws.send(
        ConversationResponseCreateEvent.from_instructions(
            "\n".join(result_instructions)
        )
    )
