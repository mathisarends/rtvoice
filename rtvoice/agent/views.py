from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel

from rtvoice.conversation.views import ConversationTurn

type OutputModality = Literal["text", "audio"]


class RealtimeModel(StrEnum):
    GPT_REALTIME = "gpt-realtime"
    GPT_REALTIME_MINI = "gpt-realtime-mini"
    GPT_REALTIME_1_5 = "gpt-realtime-1.5"


class AssistantVoice(StrEnum):
    """TTS voices available for the OpenAI Realtime API.

    Attributes:
        ALLOY: Neutral and balanced; clean output suitable for general use.
        ASH: Clear and precise; described as a male baritone with a slightly
            scratchy yet upbeat quality. May have limited performance with accents.
        BALLAD: Melodic and gentle; community notes suggest a male-sounding voice.
        CORAL: Warm and friendly; good for approachable or empathetic tones.
        ECHO: Resonant and deep; strong presence in delivery.
        FABLE: Narrative-like and expressive; fitting for storytelling contexts.
        ONYX: Darker, strong, and confident in tone.
        NOVA: Bright, youthful, and energetic.
        SAGE: Calm and thoughtful; measured pacing with a reflective quality.
        SHIMMER: Bright and energetic; dynamic expression with high clarity.
        VERSE: Versatile and expressive; adapts well across different contexts.
        CEDAR: Realtime-only voice. No official description available.
        MARIN: Realtime-only voice. No official description available.

    Example:
        ```python
        agent = RealtimeAgent(
            voice=AssistantVoice.CORAL,
        )
        ```
    """

    ALLOY = "alloy"
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SAGE = "sage"
    SHIMMER = "shimmer"
    VERSE = "verse"
    CEDAR = "cedar"
    MARIN = "marin"


class TranscriptionModel(StrEnum):
    WHISPER_1 = "whisper-1"


class NoiseReduction(StrEnum):
    NEAR_FIELD = "near_field"
    FAR_FIELD = "far_field"


class SemanticEagerness(StrEnum):
    """Controls how quickly semantic VAD decides the user has finished speaking."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class SemanticVAD(BaseModel):
    """Semantic VAD: waits for a complete thought before committing end-of-turn."""

    eagerness: SemanticEagerness = SemanticEagerness.AUTO


class ServerVAD(BaseModel):
    """Energy-based VAD: triggers end-of-turn on silence duration and audio threshold."""

    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


type TurnDetection = SemanticVAD | ServerVAD


@dataclass
class SeedMessage:
    """Pre-filled conversation message injected before live user input begins."""

    role: Literal["user", "assistant"]
    text: str

    @classmethod
    def user(cls, text: str) -> Self:
        return cls(role="user", text=text)

    @classmethod
    def assistant(cls, text: str) -> Self:
        return cls(role="assistant", text=text)


@dataclass
class ConversationSeed:
    """Conversation items sent after session.update but before mic audio starts."""

    messages: list[SeedMessage]

    @classmethod
    def from_pairs(cls, *pairs: tuple[str, str]) -> "ConversationSeed":
        messages: list[SeedMessage] = []
        for user_text, assistant_text in pairs:
            messages.append(SeedMessage.user(user_text))
            messages.append(SeedMessage.assistant(assistant_text))
        return cls(messages=messages)


@dataclass
class AgentError:
    type: str
    message: str

    def __str__(self) -> str:
        return f"[{self.type}] {self.message}"


class AgentResult(BaseModel):
    turns: list[ConversationTurn]
    recording_path: Path | None = None


@dataclass
class ClarificationCheckpoint:
    resume_history: list
    clarify_call_id: str
