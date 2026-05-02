import pytest

from rtvoice.llm import ChatInvokeUsage
from rtvoice.realtime.schemas import (
    ServerEventAdapter,
    TokenInputTokenDetails,
    TokenOutputTokenDetails,
    TokenUsage,
)
from rtvoice.token import TokenTracker


class TestTokenTracker:
    def test_tracks_chat_usage_with_cached_prompt_tokens(self) -> None:
        tracker = TokenTracker()
        usage = ChatInvokeUsage(
            prompt_tokens=100,
            prompt_cached_tokens=25,
            completion_tokens=10,
            total_tokens=110,
        )

        tracker.track_chat_usage(model="gpt-5.4-mini", usage=usage)

        summary = tracker.summary()
        assert summary.usage.input_text_tokens == 75
        assert summary.usage.cached_input_text_tokens == 25
        assert summary.usage.output_text_tokens == 10
        assert summary.cost.total_usd == pytest.approx(
            ((75 * 0.75) + (25 * 0.075) + (10 * 4.50)) / 1_000_000
        )

    def test_prices_default_openai_chat_model(self) -> None:
        tracker = TokenTracker()
        usage = ChatInvokeUsage(
            prompt_tokens=100,
            prompt_cached_tokens=25,
            completion_tokens=10,
            total_tokens=110,
        )

        tracker.track_chat_usage(model="gpt-4o", usage=usage)

        summary = tracker.summary()
        assert summary.has_unpriced_usage is False
        assert summary.cost.total_usd == pytest.approx(
            ((75 * 2.50) + (25 * 1.25) + (10 * 10.00)) / 1_000_000
        )

    def test_tracks_realtime_usage_by_modality(self) -> None:
        tracker = TokenTracker()
        usage = TokenUsage(
            input_tokens=132,
            output_tokens=121,
            total_tokens=253,
            input_token_details=TokenInputTokenDetails(
                text_tokens=119,
                audio_tokens=13,
                cached_tokens=64,
                cached_tokens_details=TokenInputTokenDetails(text_tokens=64),
            ),
            output_token_details=TokenOutputTokenDetails(
                text_tokens=30,
                audio_tokens=91,
            ),
        )

        tracker.track_realtime_response_usage(model="gpt-realtime-mini", usage=usage)

        summary = tracker.summary()
        assert summary.usage.input_text_tokens == 55
        assert summary.usage.cached_input_text_tokens == 64
        assert summary.usage.input_audio_tokens == 13
        assert summary.usage.output_audio_tokens == 91
        assert summary.cost.total_usd == pytest.approx(
            ((55 * 0.60) + (64 * 0.06) + (13 * 10.00) + (30 * 2.40) + (91 * 20.00))
            / 1_000_000
        )

    def test_marks_unknown_model_price_as_unpriced(self) -> None:
        tracker = TokenTracker()
        usage = ChatInvokeUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

        tracker.track_chat_usage(model="custom-model", usage=usage)

        summary = tracker.summary()
        assert summary.has_unpriced_usage is True
        assert summary.cost.total_usd == 0.0

    def test_marks_known_model_with_missing_token_rate_as_unpriced(self) -> None:
        tracker = TokenTracker()
        usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)

        tracker.track_transcription_usage(model="whisper-1", usage=usage)

        summary = tracker.summary()
        assert summary.has_unpriced_usage is True
        assert summary.cost.total_usd == 0.0


class TestRealtimeUsageSchema:
    def test_response_done_accepts_documented_usage_shape(self) -> None:
        event = ServerEventAdapter.validate_python(
            {
                "type": "response.done",
                "event_id": "event_1",
                "response": {
                    "id": "resp_1",
                    "usage": {
                        "total_tokens": 253,
                        "input_tokens": 132,
                        "output_tokens": 121,
                        "input_token_details": {
                            "text_tokens": 119,
                            "audio_tokens": 13,
                            "image_tokens": 0,
                            "cached_tokens": 64,
                            "cached_tokens_details": {
                                "text_tokens": 64,
                                "audio_tokens": 0,
                                "image_tokens": 0,
                            },
                        },
                        "output_token_details": {
                            "text_tokens": 30,
                            "audio_tokens": 91,
                        },
                    },
                },
            }
        )

        assert event.response.usage.total_tokens == 253
        assert event.response.usage.output_token_details.audio_tokens == 91
