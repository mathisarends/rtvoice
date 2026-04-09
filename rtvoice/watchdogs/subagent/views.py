import asyncio
from dataclasses import dataclass

from rtvoice.subagent.channel import SubAgentChannel
from rtvoice.tools.views import Tool


@dataclass
class PendingSubAgentCall:
    call_id: str
    subagent_name: str
    execution_task: asyncio.Task
    handoff_tool: Tool
    channel: SubAgentChannel
    channel_task: asyncio.Task | None = None
