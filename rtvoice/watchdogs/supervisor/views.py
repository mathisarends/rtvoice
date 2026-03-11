import asyncio
from dataclasses import dataclass, field

from rtvoice.supervisor.channel import SupervisorChannel
from rtvoice.tools.registry.views import Tool


def _preset_event() -> asyncio.Event:
    e = asyncio.Event()
    e.set()
    return e


@dataclass
class PendingSupervisorRun:
    response_id: str | None = None
    response_done: asyncio.Event = field(default_factory=_preset_event)
    pending_clarification_future: asyncio.Future[str] | None = None


@dataclass
class PendingToolCall:
    call_id: str
    tool_name: str
    result_task: asyncio.Task
    tool: Tool
    channel: SupervisorChannel
    supervisor_run: PendingSupervisorRun
    channel_task: asyncio.Task | None = None
