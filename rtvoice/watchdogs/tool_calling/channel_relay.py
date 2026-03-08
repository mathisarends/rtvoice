import logging

from rtvoice.realtime.schemas import (
    ConversationResponseCreateEvent,
    ToolChoiceMode,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.supervisor.channel import StatusMessage, UserQuestion
from rtvoice.watchdogs.tool_calling.views import PendingToolCall

logger = logging.getLogger(__name__)


class ChannelRelay:
    def __init__(self, websocket: RealtimeWebSocket) -> None:
        self._websocket = websocket

    async def run(self, pending: PendingToolCall) -> None:
        if not pending.channel:
            return

        async for event in pending.channel.events():
            if pending.supervisor_run:
                await pending.supervisor_run.response_done.wait()
                pending.supervisor_run.response_done.clear()

            if isinstance(event, StatusMessage):
                logger.debug(
                    "Supervisor status for '%s': %s", pending.tool_name, event.message
                )
                await self._send_status(event.message)
            elif isinstance(event, UserQuestion):
                logger.debug(
                    "Supervisor clarification for '%s': %s",
                    pending.tool_name,
                    event.question,
                )
                if pending.supervisor_run:
                    pending.supervisor_run.pending_clarification_future = (
                        event.answer_future
                    )
                await self._send_clarification(event.question)

    async def _send_status(self, message: str) -> None:
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                f"Briefly summarise what was done in one short natural sentence (max 12 words). "
                f"If multiple steps are listed (separated by →), combine them into one sentence. "
                f"Steps: {message}",
                tool_choice=ToolChoiceMode.NONE,
            )
        )

    async def _send_clarification(self, question: str) -> None:
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                f'Ask the user naturally and conversationally: "{question}"',
                tool_choice=ToolChoiceMode.NONE,
            )
        )
