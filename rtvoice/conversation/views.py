from dataclasses import dataclass
from typing import Literal


@dataclass
class ConversationTurn:
    role: Literal["user", "assistant"]
    transcript: str
