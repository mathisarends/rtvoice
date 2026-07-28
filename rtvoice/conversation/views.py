from dataclasses import dataclass
from typing import Literal

from rtvoice.shared.speech_speed import DEFAULT_SPEECH_SPEED, SpeechSpeed


@dataclass
class ConversationTurn:
    role: Literal["user", "assistant", "tool"]
    transcript: str
    interrupted: bool = False
    played_ms: int | None = None
    speech_speed: SpeechSpeed = DEFAULT_SPEECH_SPEED
