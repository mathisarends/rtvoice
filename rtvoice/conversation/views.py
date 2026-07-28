from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from rtvoice.shared.speech_speed import DEFAULT_SPEECH_SPEED, SpeechSpeed

_ESTIMATED_WORDS_PER_MINUTE = 150
_INTERRUPTION_MARKER = "<INTERRUPTED>"


class TurnRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class UserTurn(BaseModel):
    role: Literal[TurnRole.USER] = TurnRole.USER
    transcript: str

    def format(self) -> str:
        return f"[USER]: {self.transcript}"


class AssistantTurn(BaseModel):
    role: Literal[TurnRole.ASSISTANT] = TurnRole.ASSISTANT
    transcript: str
    interrupted: bool = False
    played_ms: int | None = None
    speech_speed: SpeechSpeed = DEFAULT_SPEECH_SPEED

    def format(self) -> str:
        if not self.interrupted:
            return f"[ASSISTANT]: {self.transcript}"

        heard_prefix = " ".join(self.transcript.split()[: self._heard_words])
        content = (
            f"{heard_prefix} {_INTERRUPTION_MARKER}"
            if heard_prefix
            else _INTERRUPTION_MARKER
        )
        return f"[ASSISTANT, INTERRUPTED]: {content}"

    @property
    def _heard_words(self) -> int:
        if self.played_ms is None:
            return 0
        return int(
            max(self.played_ms, 0)
            * self.speech_speed
            * _ESTIMATED_WORDS_PER_MINUTE
            / 60_000
        )


class ToolTurn(BaseModel):
    role: Literal[TurnRole.TOOL] = TurnRole.TOOL
    name: str
    result: str

    @property
    def transcript(self) -> str:
        return f"{self.name}: {self.result}"

    def format(self) -> str:
        return f"[TOOL]: {self.transcript}"


type ConversationTurn = UserTurn | AssistantTurn | ToolTurn
