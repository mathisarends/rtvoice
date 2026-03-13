import asyncio
from dataclasses import dataclass

from rtvoice.subagent.channel import SubAgentChannel
from rtvoice.tools.registry.views import Tool


@dataclass
class PendingToolCall:
    call_id: str
    tool_name: str
    result_task: asyncio.Task
    tool: Tool
    channel: SubAgentChannel
    channel_task: asyncio.Task | None = None
