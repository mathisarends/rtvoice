from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


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


class NoiseReduction(StrEnum):
    """
    Noise reduction filter type for audio input.

    - NEAR_FIELD: Optimized for close-range audio (e.g., headset microphone)
    - FAR_FIELD: Optimized for distant audio sources (e.g., room microphone)
    """

    NEAR_FIELD = "near_field"
    FAR_FIELD = "far_field"


class TurnDetection(BaseModel):
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


class ActionResult(BaseModel):
    success: bool = True
    message: str | None = None

    def __str__(self) -> str:
        return self.message or ("Success" if self.success else "Failed")


class TranscriptListener:
    async def on_user_chunk(self, chunk: str) -> None:
        pass

    async def on_user_completed(self, transcript: str) -> None:
        pass

    async def on_assistant_chunk(self, chunk: str) -> None:
        pass

    async def on_assistant_completed(self, transcript: str) -> None:
        pass


class AgentListener:
    async def on_agent_started(self) -> None:
        pass

    async def on_agent_stopped(self) -> None:
        pass

    async def on_agent_interrupted(self) -> None:
        pass

    async def on_subagent_called(self, agent_name: str, task: str) -> None:
        pass
