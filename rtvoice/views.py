from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel
from pydantic import Field as PydanticField

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


@dataclass(frozen=True)
class ModelPricing:
    input_text: float
    input_audio: float
    input_cached_text: float
    input_cached_audio: float
    output_text: float
    output_audio: float


REALTIME_PRICING: dict[str, ModelPricing] = {
    "gpt-realtime": ModelPricing(
        input_text=4.00,
        input_audio=32.00,
        input_cached_text=0.40,
        input_cached_audio=0.40,
        output_text=16.00,
        output_audio=64.00,
    ),
    "gpt-realtime-mini": ModelPricing(
        input_text=0.60,
        input_audio=10.00,
        input_cached_text=0.30,
        input_cached_audio=0.30,
        output_text=2.40,
        output_audio=20.00,
    ),
    "gpt-realtime-1.5": ModelPricing(
        input_text=4.00,
        input_audio=32.00,
        input_cached_text=0.40,
        input_cached_audio=0.40,
        output_text=16.00,
        output_audio=64.00,
    ),
    "gpt-4o-realtime-preview": ModelPricing(
        input_text=5.00,
        input_audio=100.00,
        input_cached_text=2.50,
        input_cached_audio=2.50,
        output_text=20.00,
        output_audio=200.00,
    ),
    "gpt-4o-mini-realtime-preview": ModelPricing(
        input_text=0.60,
        input_audio=10.00,
        input_cached_text=0.30,
        input_cached_audio=0.30,
        output_text=2.40,
        output_audio=20.00,
    ),
}


@dataclass
class TurnUsage:
    input_text_tokens: int = 0
    input_audio_tokens: int = 0
    input_cached_text_tokens: int = 0
    input_cached_audio_tokens: int = 0
    output_text_tokens: int = 0
    output_audio_tokens: int = 0

    @property
    def input_cached_tokens(self) -> int:
        return self.input_cached_text_tokens + self.input_cached_audio_tokens

    @property
    def input_tokens(self) -> int:
        return self.input_text_tokens + self.input_audio_tokens

    @property
    def output_tokens(self) -> int:
        return self.output_text_tokens + self.output_audio_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_response_done(cls, usage: Mapping[str, Any]) -> Self:
        input_details = cls._mapping(usage.get("input_token_details"))
        output_details = cls._mapping(usage.get("output_token_details"))
        cached_details = cls._mapping(input_details.get("cached_tokens_details"))
        cached_tokens = cls._int(input_details.get("cached_tokens"))
        cached_text_tokens = cls._int(cached_details.get("text_tokens"))
        cached_audio_tokens = cls._int(cached_details.get("audio_tokens"))

        if cached_tokens and not cached_text_tokens and not cached_audio_tokens:
            cached_text_tokens = cached_tokens

        return cls(
            input_text_tokens=cls._int(input_details.get("text_tokens")),
            input_audio_tokens=cls._int(input_details.get("audio_tokens")),
            input_cached_text_tokens=cached_text_tokens,
            input_cached_audio_tokens=cached_audio_tokens,
            output_text_tokens=cls._int(output_details.get("text_tokens")),
            output_audio_tokens=cls._int(output_details.get("audio_tokens")),
        )

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        return {}

    @staticmethod
    def _int(value: Any) -> int:
        if value is None:
            return 0
        return int(value)


@dataclass
class TokenUsageSummary:
    turns: list[TurnUsage] = field(default_factory=list)
    model: str = RealtimeModel.GPT_REALTIME_MINI.value

    def add(self, turn: TurnUsage) -> None:
        self.turns.append(turn)

    @property
    def total_input_text_tokens(self) -> int:
        return sum(turn.input_text_tokens for turn in self.turns)

    @property
    def total_input_audio_tokens(self) -> int:
        return sum(turn.input_audio_tokens for turn in self.turns)

    @property
    def total_input_cached_text_tokens(self) -> int:
        return sum(turn.input_cached_text_tokens for turn in self.turns)

    @property
    def total_input_cached_audio_tokens(self) -> int:
        return sum(turn.input_cached_audio_tokens for turn in self.turns)

    @property
    def total_input_cached_tokens(self) -> int:
        return (
            self.total_input_cached_text_tokens + self.total_input_cached_audio_tokens
        )

    @property
    def total_output_text_tokens(self) -> int:
        return sum(turn.output_text_tokens for turn in self.turns)

    @property
    def total_output_audio_tokens(self) -> int:
        return sum(turn.output_audio_tokens for turn in self.turns)

    @property
    def total_tokens(self) -> int:
        return sum(turn.total_tokens for turn in self.turns)

    def estimate_cost_usd(self, model: str | None = None) -> float:
        pricing_model = model or self.model
        pricing = REALTIME_PRICING.get(pricing_model)
        if not pricing:
            raise ValueError(f"No pricing data for model '{pricing_model}'")

        uncached_text_tokens = max(
            self.total_input_text_tokens - self.total_input_cached_text_tokens,
            0,
        )
        uncached_audio_tokens = max(
            self.total_input_audio_tokens - self.total_input_cached_audio_tokens,
            0,
        )
        cost = (
            (uncached_text_tokens / 1_000_000) * pricing.input_text
            + (uncached_audio_tokens / 1_000_000) * pricing.input_audio
            + (self.total_input_cached_text_tokens / 1_000_000)
            * pricing.input_cached_text
            + (self.total_input_cached_audio_tokens / 1_000_000)
            * pricing.input_cached_audio
            + (self.total_output_text_tokens / 1_000_000) * pricing.output_text
            + (self.total_output_audio_tokens / 1_000_000) * pricing.output_audio
        )
        return round(cost, 6)

    def estimate_costs_all_models(self) -> dict[str, float]:
        return {model: self.estimate_cost_usd(model) for model in REALTIME_PRICING}

    def __repr__(self) -> str:
        costs = self.estimate_costs_all_models()
        lines = [
            f"TokenUsageSummary ({len(self.turns)} turns, {self.total_tokens:,} tokens)",
            f"  input  text={self.total_input_text_tokens:,}  audio={self.total_input_audio_tokens:,}  cached={self.total_input_cached_tokens:,}",
            f"  output text={self.total_output_text_tokens:,}  audio={self.total_output_audio_tokens:,}",
            "  cost estimates:",
            *[f"    {model}: ${cost:.4f}" for model, cost in costs.items()],
        ]
        return "\n".join(lines)


class AgentResult(BaseModel):
    turns: list[ConversationTurn]
    recording_path: Path | None = None
    usage: TokenUsageSummary = PydanticField(default_factory=TokenUsageSummary)


@dataclass
class ClarificationCheckpoint:
    resume_history: list
    clarify_call_id: str
