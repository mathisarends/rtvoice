import asyncio
from dataclasses import dataclass

from rtvoice.supervisor.channel import SupervisorChannel
from rtvoice.tools.registry.views import Tool


def _preset_event() -> asyncio.Event:
    e = asyncio.Event()
    e.set()
    return e


@dataclass
class PendingToolCall:
    call_id: str
    tool_name: str
    result_task: asyncio.Task
    tool: Tool
    channel: SupervisorChannel
    channel_task: asyncio.Task | None = None
