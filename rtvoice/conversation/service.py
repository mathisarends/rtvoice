from rtvoice.conversation.views import ConversationTurn
from rtvoice.events import EventBus
from rtvoice.events.views import (
    AssistantTranscriptCompletedEvent,
    UserTranscriptCompletedEvent,
)


class ConversationHistory:
    def __init__(self, event_bus: EventBus, user_transcription_enabled: bool = False):
        self._turns: list[ConversationTurn] = []
        self._user_transcription_enabled = user_transcription_enabled
        event_bus.subscribe(UserTranscriptCompletedEvent, self._on_user)
        event_bus.subscribe(AssistantTranscriptCompletedEvent, self._on_assistant)

    async def _on_user(self, event: UserTranscriptCompletedEvent) -> None:
        self._turns.append(ConversationTurn(role="user", transcript=event.transcript))

    async def _on_assistant(self, event: AssistantTranscriptCompletedEvent) -> None:
        self._turns.append(
            ConversationTurn(role="assistant", transcript=event.transcript)
        )

    @property
    def turns(self) -> list[ConversationTurn]:
        return list(self._turns)

    def format(self) -> str:
        if not self._turns:
            return "(no conversation yet)"

        lines = []
        for i, turn in enumerate(self._turns):
            if self._needs_user_placeholder(i, turn):
                lines.append("[USER]: (not recorded)")
            lines.append(f"[{turn.role.upper()}]: {turn.transcript}")
        return "\n".join(lines)

    def _needs_user_placeholder(self, index: int, turn: ConversationTurn) -> bool:
        if turn.role != "assistant":
            return False
        if self._user_transcription_enabled:
            return False

        preceding_turn = self._turns[index - 1] if index > 0 else None
        preceding_is_user = preceding_turn is not None and preceding_turn.role == "user"
        return not preceding_is_user
