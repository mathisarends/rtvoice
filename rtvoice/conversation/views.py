from dataclasses import dataclass
from typing import Literal


@dataclass
class ConversationTurn:
    role: Literal["user", "assistant"]
    transcript: str
    interrupted: bool = False
    played_ms: int | None = None
    speech_speed: float = 1.0
