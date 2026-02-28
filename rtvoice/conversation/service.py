from rtvoice.conversation.views import ConversationTurn
from rtvoice.events import EventBus
from rtvoice.events.views import (
    AssistantTranscriptCompletedEvent,
    UserTranscriptCompletedEvent,
)


class ConversationHistory:
    def __init__(self, event_bus: EventBus):
        self._turns: list[ConversationTurn] = []
        event_bus.subscribe(UserTranscriptCompletedEvent, self._on_user)
        event_bus.subscribe(AssistantTranscriptCompletedEvent, self._on_assistant)

    async def _on_user(self, event: UserTranscriptCompletedEvent) -> None:
        self._turns.append(ConversationTurn(role="user", transcript=event.transcript))

    async def _on_assistant(self, event: AssistantTranscriptCompletedEvent) -> None:
        self._turns.append(
            ConversationTurn(role="assistant", transcript=event.transcript)
        )

    def format(self) -> str:
        if not self._turns:
            return "(no conversation yet)"

        return "\n".join(
            f"[{turn.role.upper()}]: {turn.transcript}" for turn in self._turns
        )
