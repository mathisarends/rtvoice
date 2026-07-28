import pytest
from pydantic import ValidationError

from rtvoice.agent.views import (
    AssistantVoice,
    NoiseReduction,
    RealtimeModel,
    ReasoningEffort,
    SemanticEagerness,
    SemanticVAD,
    ServerVAD,
    TranscriptionModel,
)
from rtvoice.realtime.session_settings import (
    RealtimeSessionSettings,
    build_session_payload,
)


def payload(settings: RealtimeSessionSettings) -> dict:
    return build_session_payload(settings, tools=[]).model_dump(
        exclude_none=True, mode="json"
    )


class TestSettingsDefaults:
    def test_output_modalities_drop_duplicates(self) -> None:
        settings = RealtimeSessionSettings(output_modalities=("audio", "text", "audio"))
        assert settings.output_modalities == ("audio", "text")

    def test_assistant_text_enabled_follows_modalities(self) -> None:
        assert not RealtimeSessionSettings().assistant_text_enabled
        assert RealtimeSessionSettings(
            output_modalities=("audio", "text")
        ).assistant_text_enabled

    def test_transcription_enabled_follows_model(self) -> None:
        assert RealtimeSessionSettings().transcription_enabled
        assert not RealtimeSessionSettings(
            transcription_model=None
        ).transcription_enabled

    def test_settings_are_frozen(self) -> None:
        settings = RealtimeSessionSettings()
        with pytest.raises(ValidationError):
            settings.model = RealtimeModel.GPT_REALTIME_2

    def test_speech_speed_is_clamped_to_api_range(self) -> None:
        assert RealtimeSessionSettings(speech_speed=9.0).speech_speed == 1.5


class TestWirePayload:
    def test_full_payload_shape(self) -> None:
        settings = RealtimeSessionSettings(
            model=RealtimeModel.GPT_REALTIME_2_1_MINI,
            reasoning_effort=ReasoningEffort.LOW,
            instructions="Be concise.",
            voice=AssistantVoice.MARIN,
            speech_speed=1.2,
            transcription_model=TranscriptionModel.WHISPER_1,
            output_modalities=("audio",),
            noise_reduction=NoiseReduction.FAR_FIELD,
            turn_detection=SemanticVAD(eagerness=SemanticEagerness.HIGH),
        )

        assert payload(settings) == {
            "type": "realtime",
            "model": "gpt-realtime-2.1-mini",
            "reasoning": {"effort": "low"},
            "instructions": "Be concise.",
            "max_output_tokens": "inf",
            "output_modalities": ["audio"],
            "tool_choice": "auto",
            "tools": [],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {
                        "type": "semantic_vad",
                        "eagerness": "high",
                        "create_response": True,
                        "interrupt_response": True,
                    },
                    "transcription": {"model": "whisper-1"},
                    "noise_reduction": {"type": "far_field"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "speed": 1.2,
                    "voice": "marin",
                },
            },
        }

    def test_server_vad_maps_thresholds(self) -> None:
        settings = RealtimeSessionSettings(
            turn_detection=ServerVAD(
                threshold=0.8, prefix_padding_ms=100, silence_duration_ms=900
            )
        )

        assert payload(settings)["audio"]["input"]["turn_detection"] == {
            "type": "server_vad",
            "threshold": 0.8,
            "prefix_padding_ms": 100,
            "silence_duration_ms": 900,
            "create_response": True,
            "interrupt_response": True,
        }

    def test_near_field_noise_reduction_maps_to_wire_value(self) -> None:
        settings = RealtimeSessionSettings(noise_reduction=NoiseReduction.NEAR_FIELD)

        assert payload(settings)["audio"]["input"]["noise_reduction"] == {
            "type": "near_field"
        }

    def test_reasoning_omitted_when_effort_is_none(self) -> None:
        assert "reasoning" not in payload(
            RealtimeSessionSettings(reasoning_effort=None)
        )

    def test_transcription_omitted_when_model_is_none(self) -> None:
        settings = RealtimeSessionSettings(transcription_model=None)

        assert "transcription" not in payload(settings)["audio"]["input"]

    def test_tools_reach_the_wire(self) -> None:
        from rtvoice.realtime.schemas import FunctionParameters, FunctionTool

        tool = FunctionTool(name="get_time", parameters=FunctionParameters())
        wire = build_session_payload(RealtimeSessionSettings(), tools=[tool])

        assert wire.tools == [tool]
