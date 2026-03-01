import logging

from rtvoice.events import EventBus
from rtvoice.events.views import UserTranscriptCompletedEvent
from rtvoice.realtime.schemas import (
    ConversationResponseCreateEvent,
    ToolChoiceMode,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.subagents.views import SubAgentClarificationNeeded

logger = logging.getLogger(__name__)


class SubAgentInteractionWatchdog:
    def __init__(self, event_bus: EventBus, websocket: RealtimeWebSocket):
        self._event_bus = event_bus
        self._websocket = websocket
        self._pending_clarification: SubAgentClarificationNeeded | None = None

        self._event_bus.subscribe(
            SubAgentClarificationNeeded, self._on_clarification_needed
        )
        self._event_bus.subscribe(
            UserTranscriptCompletedEvent, self._on_user_transcript
        )

    async def _on_clarification_needed(
        self, event: SubAgentClarificationNeeded
    ) -> None:
        logger.debug("Clarification needed: %s", event.question)
        self._pending_clarification = event

        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                f'Ask the user naturally and conversationally: "{event.question}"',
                tool_choice=ToolChoiceMode.NONE,
            )
        )

    async def _on_user_transcript(self, event: UserTranscriptCompletedEvent) -> None:
        if self._pending_clarification is None:
            return

        clarification = self._pending_clarification
        self._pending_clarification = None

        logger.debug("Clarification answered: %s", event.transcript)
        clarification.answer_future.set_result(event.transcript)
