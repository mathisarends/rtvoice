from decimal import Decimal

import pytest
from transitbus import EventBus

from rtvoice.realtime.schemas import (
    DurationUsage,
    InputAudioTranscriptionCompleted,
    RealtimeResponseObject,
    ResponseDoneEvent,
    TokenInputTokenDetails,
    TokenOutputTokenDetails,
    TokenUsage,
)
from rtvoice.tokens import Currency, TokenTracker


def response_done() -> ResponseDoneEvent:
    return ResponseDoneEvent(
        type="response.done",
        event_id="event-1",
        response=RealtimeResponseObject(
            id="response-1",
            usage=TokenUsage(
                input_tokens=155,
                output_tokens=50,
                total_tokens=205,
                input_token_details=TokenInputTokenDetails(
                    text_tokens=100,
                    audio_tokens=50,
                    image_tokens=5,
                    cached_tokens=50,
                    cached_tokens_details=TokenInputTokenDetails(
                        text_tokens=40,
                        audio_tokens=10,
                        image_tokens=0,
                    ),
                ),
                output_token_details=TokenOutputTokenDetails(
                    text_tokens=20,
                    audio_tokens=30,
                ),
            ),
        ),
    )


def transcription_done() -> InputAudioTranscriptionCompleted:
    return InputAudioTranscriptionCompleted(
        type="conversation.item.input_audio_transcription.completed",
        event_id="event-2",
        item_id="item-1",
        content_index=0,
        transcript="hello",
        usage=TokenUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_token_details=TokenInputTokenDetails(
                text_tokens=0,
                audio_tokens=100,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_tracks_response_and_transcription_events() -> None:
    event_bus = EventBus()
    tracker = TokenTracker(
        event_bus=event_bus,
        realtime_model="gpt-realtime-2.1",
        transcription_model="whisper-1",
    )

    await event_bus.dispatch(response_done())
    await event_bus.dispatch(transcription_done())

    report = tracker.report()
    assert report.tokens.realtime.input_audio_tokens == 50
    assert report.tokens.realtime.cached_input_text_tokens == 40
    assert report.tokens.realtime.output_audio_tokens == 30
    assert report.tokens.transcription.input_audio_tokens == 100
    assert report.cost.currency == "USD"
    assert report.cost.total == Decimal("0.004965")
    assert report.cost.is_complete
    assert "100 ms per audio token" in report.cost.notes[0]


@pytest.mark.asyncio
async def test_deduplicates_repeated_events() -> None:
    event_bus = EventBus()
    tracker = TokenTracker(
        event_bus=event_bus,
        realtime_model="gpt-realtime-2.1",
        transcription_model="whisper-1",
    )
    event = response_done()
    transcription = transcription_done()

    await event_bus.dispatch(event)
    await event_bus.dispatch(event)
    await event_bus.dispatch(transcription)
    await event_bus.dispatch(transcription)

    assert tracker.totals.realtime.responses == 1
    assert tracker.totals.transcription.transcriptions == 1


def test_native_currency_is_usd() -> None:
    assert list(Currency) == [Currency.USD]
    assert Currency.USD == "USD"


def test_keeps_event_bus_reference() -> None:
    event_bus = EventBus()

    tracker = TokenTracker(
        event_bus=event_bus,
        realtime_model="gpt-realtime-2.1",
    )

    assert tracker._event_bus is event_bus


@pytest.mark.asyncio
async def test_prices_duration_based_whisper_usage() -> None:
    event_bus = EventBus()
    tracker = TokenTracker(
        event_bus=event_bus,
        realtime_model="gpt-realtime-2.1",
        transcription_model="whisper-1",
    )
    event = transcription_done().model_copy(update={"usage": DurationUsage(seconds=30)})

    await event_bus.dispatch(event)

    report = tracker.report()
    assert report.cost.total == Decimal("0.003")
    assert report.cost.line_items[0].rate_unit == "minute"
    assert not report.cost.notes


@pytest.mark.asyncio
async def test_prices_token_based_transcription_model() -> None:
    event_bus = EventBus()
    tracker = TokenTracker(
        event_bus=event_bus,
        realtime_model="gpt-realtime-2.1",
        transcription_model="gpt-4o-transcribe",
    )

    await event_bus.dispatch(transcription_done())

    report = tracker.report()
    assert report.cost.total == Decimal("0.00045")
    assert {item.category for item in report.cost.line_items} == {
        "transcription.input",
        "transcription.output",
    }


@pytest.mark.asyncio
async def test_marks_estimate_incomplete_without_modality_details() -> None:
    event_bus = EventBus()
    tracker = TokenTracker(
        event_bus=event_bus,
        realtime_model="gpt-realtime-2.1",
    )
    event = response_done()
    event.response.usage = TokenUsage(
        input_tokens=10,
        output_tokens=0,
        total_tokens=10,
    )

    await event_bus.dispatch(event)

    report = tracker.report()
    assert report.cost.total == 0
    assert not report.cost.is_complete
    assert "could not be assigned" in report.cost.notes[0]
