from __future__ import annotations

from typing import assert_never

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rtvoice.agent.views import (
    AssistantVoice,
    NoiseReduction,
    OutputModality,
    RealtimeModel,
    ReasoningEffort,
    SemanticVAD,
    ServerVAD,
    TranscriptionModel,
    TurnDetection,
)
from rtvoice.realtime.schemas import (
    AudioInputSettings,
    AudioOutputSettings,
    AudioSettings,
    FunctionTool,
    InputAudioNoiseReductionSettings,
    InputAudioTranscriptionSettings,
    RealtimeSessionPayload,
    ReasoningSettings,
    SemanticVADSettings,
    ServerVADSettings,
    ToolChoiceMode,
    TurnDetectionSettings,
)
from rtvoice.shared.speech_speed import DEFAULT_SPEECH_SPEED, SpeechSpeed


class RealtimeSessionSettings(BaseModel):
    """Everything about a session that ends up in `session.update`, in the
    caller-facing vocabulary. Frozen so it cannot drift from what was sent."""

    model_config = ConfigDict(frozen=True)

    model: RealtimeModel = RealtimeModel.GPT_REALTIME_2_1_MINI
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.LOW
    instructions: str = ""
    voice: AssistantVoice = AssistantVoice.MARIN
    speech_speed: SpeechSpeed = DEFAULT_SPEECH_SPEED
    transcription_model: TranscriptionModel | None = TranscriptionModel.WHISPER_1
    output_modalities: tuple[OutputModality, ...] = ("audio",)
    noise_reduction: NoiseReduction = NoiseReduction.FAR_FIELD
    turn_detection: TurnDetection = Field(default_factory=SemanticVAD)

    @field_validator("output_modalities", mode="after")
    @classmethod
    def _drop_duplicates(
        cls, value: tuple[OutputModality, ...]
    ) -> tuple[OutputModality, ...]:
        return tuple(dict.fromkeys(value))

    @property
    def assistant_text_enabled(self) -> bool:
        return "text" in self.output_modalities

    @property
    def transcription_enabled(self) -> bool:
        return self.transcription_model is not None

    @property
    def summary(self) -> str:
        return (
            f"model={self.model}, reasoning_effort={self.reasoning_effort}, "
            f"voice={self.voice}, speed={self.speech_speed}, "
            f"turn_detection={type(self.turn_detection).__name__}, "
            f"transcription={self.transcription_model}, "
            f"output_modalities={list(self.output_modalities)}"
        )


def build_session_payload(
    settings: RealtimeSessionSettings, tools: list[FunctionTool]
) -> RealtimeSessionPayload:
    return RealtimeSessionPayload(
        model=settings.model,
        reasoning=_reasoning(settings.reasoning_effort),
        instructions=settings.instructions,
        output_modalities=list(settings.output_modalities),
        tool_choice=ToolChoiceMode.AUTO,
        tools=tools,
        audio=AudioSettings(
            input=AudioInputSettings(
                turn_detection=_turn_detection(settings.turn_detection),
                noise_reduction=InputAudioNoiseReductionSettings(
                    type=settings.noise_reduction
                ),
                transcription=_transcription(settings.transcription_model),
            ),
            output=AudioOutputSettings(
                voice=settings.voice.value, speed=settings.speech_speed
            ),
        ),
    )


def _reasoning(effort: ReasoningEffort | None) -> ReasoningSettings | None:
    return None if effort is None else ReasoningSettings(effort=effort)


def _transcription(
    model: TranscriptionModel | None,
) -> InputAudioTranscriptionSettings | None:
    return None if model is None else InputAudioTranscriptionSettings(model=model)


def _turn_detection(turn_detection: TurnDetection) -> TurnDetectionSettings:
    match turn_detection:
        case SemanticVAD(eagerness=eagerness):
            return SemanticVADSettings(eagerness=eagerness)
        case ServerVAD(
            threshold=threshold,
            prefix_padding_ms=prefix_padding_ms,
            silence_duration_ms=silence_duration_ms,
        ):
            return ServerVADSettings(
                threshold=threshold,
                prefix_padding_ms=prefix_padding_ms,
                silence_duration_ms=silence_duration_ms,
            )
        case _:
            assert_never(turn_detection)
