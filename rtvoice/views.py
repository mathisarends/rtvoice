from enum import StrEnum

from pydantic import BaseModel, field_validator


# TODO: Restrucure this here
class RealtimeModel(StrEnum):
    GPT_REALTIME = "gpt-realtime"
    GPT_REALTIME_MINI = "gpt-realtime-mini"


class AssistantVoice(StrEnum):
    """
    Available assistant voices for the OpenAI Realtime API.

    Each voice has distinct characteristics suited for different use-cases
    such as narration, conversational dialogue, or expressive responses.

    - alloy: Neutral and balanced, clean output suitable for general use.
    - ash: Clear and precise; described as a male baritone with a slightly
      scratchy yet upbeat quality. May have limited performance with accents.
    - ballad: Melodic and gentle; community notes suggest a male-sounding voice.
    - coral: Warm and friendly, good for approachable or empathetic tones.
    - echo: Resonant and deep, strong presence in delivery.
    - fable: Not officially documented; often perceived as narrative-like
      and expressive, fitting for storytelling contexts.
    - onyx: Not officially documented; often perceived as darker, strong,
      and confident in tone.
    - nova: Not officially documented; frequently described as bright,
      youthful, or energetic.
    - sage: Calm and thoughtful, measured pacing and a reflective quality.
    - shimmer: Bright and energetic, dynamic expression with high clarity.
    - verse: Versatile and expressive, adapts well across different contexts.
    - cedar: (Realtime-only) - no official description available yet.
    - marin: (Realtime-only) - no official description available yet.
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


class NoiseReductionType(StrEnum):
    NEAR_FIELD = "near_field"
    FAR_FIELD = "far_field"


class TranscriptionSettings(BaseModel):
    enabled: bool = False
    model: TranscriptionModel = TranscriptionModel.WHISPER_1
    language: str | None = None
    prompt: str | None = None
    noise_reduction_mode: NoiseReductionType | None = None

    @field_validator("language")
    @classmethod
    def validate_language_code(cls, value: str | None) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError("Language code must be a string")

        lang = value.strip().lower()
        if not lang:
            return None

        if len(lang) in (2, 3) and lang.isalpha():
            return lang

        raise ValueError(
            f"Invalid language code: {value!r}. Expected ISO-639-1 format (e.g., 'en', 'de')"
        )
