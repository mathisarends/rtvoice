import logging

from rtvoice.realtime.schemas import (
    ConversationResponseCreateEvent,
    ToolChoiceMode,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.watchdogs.supervisor.views import PendingToolCall

logger = logging.getLogger(__name__)


class ChannelRelay:
    def __init__(self, websocket: RealtimeWebSocket) -> None:
        self._websocket = websocket

    async def run(self, pending: PendingToolCall) -> None:
        if not pending.channel:
            return

        async for event in pending.channel.events():
            logger.debug(
                "Channel event received for '%s': %s",
                pending.tool_name,
                type(event).__name__,
            )

            await pending.holding_done.wait()
            pending.holding_done.clear()

            logger.debug(
                "Supervisor status for '%s': %s", pending.tool_name, event.message
            )
            await self._send_status(event.message)

        logger.debug("Channel events exhausted for '%s'", pending.tool_name)

    async def _send_status(self, message: str) -> None:
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                f"Briefly summarise what was done in one short natural sentence (max 12 words). "
                f"If multiple steps are listed (separated by →), combine them into one sentence. "
                f"Steps: {message}",
                tool_choice=ToolChoiceMode.NONE,
            )
        )
