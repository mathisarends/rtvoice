import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentStoppedEvent,
    AssistantTranscriptCompletedEvent,
    ConversationHistoryResponseEvent,
    UserTranscriptCompletedEvent,
)


@dataclass
class ConversationTurn:
    role: Literal["user", "assistant"]
    transcript: str
    item_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    output_index: int | None = None
    content_index: int | None = None


logger = logging.getLogger(__name__)


class ConversationHistoryWatchdog:
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._conversation_history: list[ConversationTurn] = []

        self._event_bus.subscribe(
            UserTranscriptCompletedEvent,
            self._on_user_transcript_completed,
        )
        self._event_bus.subscribe(
            AssistantTranscriptCompletedEvent,
            self._on_assistant_transcript_completed,
        )
        self._event_bus.subscribe(
            AgentStoppedEvent,
            self._on_agent_stopped,
        )

    @property
    def conversation_history(self) -> list[ConversationTurn]:
        return self._conversation_history.copy()

    def clear_history(self) -> None:
        self._conversation_history.clear()
        logger.info("Conversation history cleared")

    async def _on_user_transcript_completed(
        self, event: UserTranscriptCompletedEvent
    ) -> None:
        turn = ConversationTurn(
            role="user",
            transcript=event.transcript,
            item_id=event.item_id,
        )
        self._conversation_history.append(turn)
        logger.debug("Added user turn to conversation history")

    async def _on_assistant_transcript_completed(
        self, event: AssistantTranscriptCompletedEvent
    ) -> None:
        turn = ConversationTurn(
            role="assistant",
            transcript=event.transcript,
            item_id=event.item_id,
            output_index=event.output_index,
            content_index=event.content_index,
        )
        self._conversation_history.append(turn)
        logger.debug("Added assistant turn to conversation history")

    async def _on_agent_stopped(self, event: AgentStoppedEvent) -> None:
        logger.info(
            "Agent stopped - publishing conversation history (%d turns)",
            len(self._conversation_history),
        )

        response_event = ConversationHistoryResponseEvent(
            conversation_turns=self._conversation_history.copy()
        )
        await self._event_bus.dispatch(response_event)
