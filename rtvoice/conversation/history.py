from collections.abc import Iterable

from transitbus import EventBus

from rtvoice.conversation.views import ConversationTurn
from rtvoice.events.views import (
    AssistantTranscriptCompletedEvent,
    UserTranscriptCompletedEvent,
)


class ConversationHistory:
    def __init__(self, event_bus: EventBus):
        self._turns: list[ConversationTurn] = []

        event_bus.on(UserTranscriptCompletedEvent, self._on_user)
        event_bus.on(AssistantTranscriptCompletedEvent, self._on_assistant)

    async def _on_user(self, event: UserTranscriptCompletedEvent) -> None:
        self._turns.append(ConversationTurn(role="user", transcript=event.transcript))

    async def _on_assistant(self, event: AssistantTranscriptCompletedEvent) -> None:
        self._turns.append(
            ConversationTurn(role="assistant", transcript=event.transcript)
        )

    def add(self, turn: ConversationTurn) -> None:
        self._turns.append(turn)

    def seed(self, turns: Iterable[ConversationTurn]) -> None:
        self._turns.extend(turns)

    @property
    def turns(self) -> list[ConversationTurn]:
        return list(self._turns)

    def format(self) -> str:
        if not self._turns:
            return "(no conversation yet)"

        lines = [f"[{turn.role.upper()}]: {turn.transcript}" for turn in self._turns]
        return "\n".join(lines)
