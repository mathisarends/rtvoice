from collections.abc import Iterable
from dataclasses import dataclass, replace

from transitbus import EventBus

from rtvoice.conversation.views import ConversationTurn
from rtvoice.events.views import (
    AssistantInterruptedEvent,
    AssistantTranscriptCompletedEvent,
    UserTranscriptCompletedEvent,
)
from rtvoice.shared.speech_speed import DEFAULT_SPEECH_SPEED, SpeechSpeed

_ESTIMATED_WORDS_PER_MINUTE = 150
_INTERRUPTION_MARKER = "<INTERRUPTED>"

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


def _estimated_heard_words(turn: ConversationTurn) -> int:
    if turn.played_ms is None:
        return 0
    return int(
        max(turn.played_ms, 0)
        * turn.speech_speed
        * _ESTIMATED_WORDS_PER_MINUTE
        / 60_000
    )


def _format_turn(turn: ConversationTurn) -> str:
    if turn.role != "assistant" or not turn.interrupted:
        return f"[{turn.role.upper()}]: {turn.transcript}"

    heard_words = _estimated_heard_words(turn)
    heard_prefix = " ".join(turn.transcript.split()[:heard_words])
    content = (
        f"{heard_prefix} {_INTERRUPTION_MARKER}"
        if heard_prefix
        else _INTERRUPTION_MARKER
    )
    return f"[ASSISTANT, INTERRUPTED]: {content}"


class ConversationHistory:
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._turns: list[ConversationTurn] = []
        self._assistant_turns: dict[_AssistantTurnKey, _TurnIndex] = {}
        self._pending_interruptions: dict[_AssistantTurnKey, _Interruption] = {}

        self._event_bus.on(UserTranscriptCompletedEvent, self._on_user)
        self._event_bus.on(AssistantTranscriptCompletedEvent, self._on_assistant)
        self._event_bus.on(AssistantInterruptedEvent, self._on_assistant_interrupted)

    async def _on_user(self, event: UserTranscriptCompletedEvent) -> None:
        self._turns.append(ConversationTurn(role="user", transcript=event.transcript))

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
            ConversationTurn(
                role="assistant",
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
                self._turns[turn_index] = replace(
                    self._turns[turn_index],
                    interrupted=True,
                    played_ms=event.played_ms,
                    speech_speed=event.speech_speed,
                )
            return
        if keys:
            self._pending_interruptions.update(dict.fromkeys(keys, interruption))
            return

        for turn_index in range(len(self._turns) - 1, -1, -1):
            if self._turns[turn_index].role == "assistant":
                self._turns[turn_index] = replace(
                    self._turns[turn_index],
                    interrupted=True,
                    played_ms=event.played_ms,
                    speech_speed=event.speech_speed,
                )
                return

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

        lines = [_format_turn(turn) for turn in self._turns]
        return "\n".join(lines)
