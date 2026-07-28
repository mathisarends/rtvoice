import warnings
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from rtvoice.conversation.views import ConversationTurn
from rtvoice.tokens.models import UsageReport

type OutputModality = Literal["text", "audio"]


class RealtimeModel(StrEnum):
    GPT_REALTIME_2_1 = "gpt-realtime-2.1"
    GPT_REALTIME_2_1_MINI = "gpt-realtime-2.1-mini"
    GPT_REALTIME_2 = "gpt-realtime-2"
    GPT_REALTIME_1_5 = "gpt-realtime-1.5"
    GPT_REALTIME = "gpt-realtime"
    GPT_REALTIME_MINI = "gpt-realtime-mini"

    def warn_if_deprecated(self, *, stacklevel: int = 2) -> None:
        replacement = _DEPRECATED_REALTIME_MODELS.get(self)
        if replacement is None:
            return
        warnings.warn(
            f"{self.value!r} is deprecated and scheduled for shutdown on "
            f"2027-01-20; use {replacement.value!r} instead.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )


_DEPRECATED_REALTIME_MODELS = {
    RealtimeModel.GPT_REALTIME: RealtimeModel.GPT_REALTIME_2_1,
    RealtimeModel.GPT_REALTIME_MINI: RealtimeModel.GPT_REALTIME_2_1_MINI,
}


class ReasoningEffort(StrEnum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class AssistantVoice(StrEnum):
    """
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
    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"


class NoiseReduction(StrEnum):
    NEAR_FIELD = "near_field"
    FAR_FIELD = "far_field"


class SemanticEagerness(StrEnum):
    """Controls how quickly semantic VAD decides the user has finished speaking."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class TurnDetectionType(StrEnum):
    SEMANTIC_VAD = "semantic_vad"
    SERVER_VAD = "server_vad"


class SemanticVAD(BaseModel):
    """Semantic VAD: waits for a complete thought before committing end-of-turn."""

    type: Literal[TurnDetectionType.SEMANTIC_VAD] = TurnDetectionType.SEMANTIC_VAD
    eagerness: SemanticEagerness = SemanticEagerness.AUTO


class ServerVAD(BaseModel):
    """Energy-based VAD: triggers end-of-turn on silence duration and audio threshold."""

    type: Literal[TurnDetectionType.SERVER_VAD] = TurnDetectionType.SERVER_VAD
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


TurnDetection = Annotated[SemanticVAD | ServerVAD, Field(discriminator="type")]


class InjectedUserMessage(BaseModel):
    """Pre-filled user message injected before live user input begins."""

    text: str


class InjectedAssistantMessage(BaseModel):
    """Pre-filled assistant message injected before live user input begins."""

    text: str


class InjectedConversation(BaseModel):
    """Conversation items sent after session.update but before mic audio starts."""

    messages: list[InjectedUserMessage | InjectedAssistantMessage]


class AgentError(BaseModel):
    type: str
    message: str

    def __str__(self) -> str:
        return f"[{self.type}] {self.message}"


class AgentResult(BaseModel):
    turns: list[ConversationTurn]
    recording_path: Path | None = None
    usage: UsageReport
