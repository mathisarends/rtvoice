from collections.abc import Iterable
from dataclasses import dataclass

from transitbus import EventBus

from rtvoice.conversation.views import (
    AssistantTurn,
    ConversationTurn,
    ToolTurn,
    UserTurn,
)
from rtvoice.events.views import (
    AssistantInterruptedEvent,
    AssistantTranscriptCompletedEvent,
    ToolExecutedEvent,
    UserTranscriptCompletedEvent,
)
from rtvoice.shared.speech_speed import DEFAULT_SPEECH_SPEED, SpeechSpeed

type _AssistantTurnKey = str
type _TurnIndex = int


@dataclass(frozen=True)
class _Interruption:
    played_ms: int | None
    speech_speed: SpeechSpeed


def _assistant_keys(item_id: str, response_id: str | None) -> set[_AssistantTurnKey]:
    keys: set[_AssistantTurnKey] = {f"item:{item_id}"}
    if response_id:
        keys.add(f"response:{response_id}")
    return keys


def _interruption_keys(
    item_id: str | None, response_id: str | None
) -> set[_AssistantTurnKey]:
    if item_id:
        return {f"item:{item_id}"}
    if response_id:
        return {f"response:{response_id}"}
    return set()


class ConversationHistory:
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._turns: list[ConversationTurn] = []
        self._assistant_turns: dict[_AssistantTurnKey, _TurnIndex] = {}
        self._pending_interruptions: dict[_AssistantTurnKey, _Interruption] = {}

        self._event_bus.on(UserTranscriptCompletedEvent, self._on_user)
        self._event_bus.on(AssistantTranscriptCompletedEvent, self._on_assistant)
        self._event_bus.on(AssistantInterruptedEvent, self._on_assistant_interrupted)
        self._event_bus.on(ToolExecutedEvent, self._on_tool_executed)

    async def _on_user(self, event: UserTranscriptCompletedEvent) -> None:
        self._turns.append(UserTurn(transcript=event.transcript))

    async def _on_tool_executed(self, event: ToolExecutedEvent) -> None:
        self._turns.append(ToolTurn(name=event.name, result=event.result))

    async def _on_assistant(self, event: AssistantTranscriptCompletedEvent) -> None:
        keys = _assistant_keys(event.item_id, event.response_id)
        interruption = next(
            (
                self._pending_interruptions[key]
                for key in keys
                if key in self._pending_interruptions
            ),
            None,
        )
        turn_index: _TurnIndex = len(self._turns)
        self._turns.append(
            AssistantTurn(
                transcript=event.transcript,
                interrupted=interruption is not None,
                played_ms=interruption.played_ms if interruption else None,
                speech_speed=(
                    interruption.speech_speed if interruption else DEFAULT_SPEECH_SPEED
                ),
            )
        )
        self._assistant_turns.update(dict.fromkeys(keys, turn_index))
        for key in keys:
            self._pending_interruptions.pop(key, None)

    async def _on_assistant_interrupted(self, event: AssistantInterruptedEvent) -> None:
        keys = _interruption_keys(event.item_id, event.response_id)
        interruption = _Interruption(event.played_ms, event.speech_speed)
        turn_indices: set[_TurnIndex] = {
            self._assistant_turns[key] for key in keys if key in self._assistant_turns
        }
        if turn_indices:
            for turn_index in turn_indices:
                self._mark_interrupted(turn_index, interruption)
            return
        if keys:
            self._pending_interruptions.update(dict.fromkeys(keys, interruption))
            return

        for turn_index in range(len(self._turns) - 1, -1, -1):
            if isinstance(self._turns[turn_index], AssistantTurn):
                self._mark_interrupted(turn_index, interruption)
                return

    def _mark_interrupted(
        self, turn_index: _TurnIndex, interruption: _Interruption
    ) -> None:
        turn = self._turns[turn_index]
        if not isinstance(turn, AssistantTurn):
            return

        self._turns[turn_index] = turn.model_copy(
            update={
                "interrupted": True,
                "played_ms": interruption.played_ms,
                "speech_speed": interruption.speech_speed,
            }
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

        lines = [turn.format() for turn in self._turns]
        return "\n".join(lines)
