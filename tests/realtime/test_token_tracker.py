from decimal import Decimal

import pytest
from tokenary import ModelName
from tokenary.generator.schemas import GeneratedModelPricing
from transitbus import EventBus

import rtvoice.tokens.pricing as pricing_module
from rtvoice.realtime.schemas import (
    DurationUsage,
    InputAudioTranscriptionCompleted,
    RealtimeResponseObject,
    ResponseDoneEvent,
    TokenInputTokenDetails,
    TokenOutputTokenDetails,
    TokenUsage,
)
from rtvoice.tokens import (
    Currency,
    PricingCatalog,
    RealtimeRates,
    RealtimeTokenTotals,
    TokenTotals,
    TokenTracker,
)

REALTIME_MODELS = [
    "gpt-realtime",
    "gpt-realtime-1.5",
    "gpt-realtime-2",
    "gpt-realtime-2.1",
    "gpt-realtime-2.1-mini",
    "gpt-realtime-2025-08-28",
    "gpt-realtime-mini",
    "gpt-realtime-mini-2025-10-06",
    "gpt-realtime-mini-2025-12-15",
]


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
async def test_prices_dated_realtime_model_from_tokenary() -> None:
    event_bus = EventBus()
    tracker = TokenTracker(
        event_bus=event_bus,
        realtime_model="gpt-realtime-mini-2025-12-15",
    )

    await event_bus.dispatch(response_done())

    report = tracker.report()
    assert report.cost.total == Decimal("0.0010934")
    assert report.cost.pricing_source.startswith("tokenary ")
    assert report.cost.is_complete


@pytest.mark.parametrize("model", REALTIME_MODELS)
def test_tokenary_realtime_model_is_configured(model: str) -> None:
    totals = TokenTotals(
        realtime=RealtimeTokenTotals(
            responses=1,
            total_tokens=2,
            input_tokens=1,
            output_tokens=1,
            input_text_tokens=1,
            output_text_tokens=1,
        )
    )

    estimate = PricingCatalog().estimate(
        totals,
        realtime_model=model,
        transcription_model=None,
    )

    assert estimate.total > 0
    assert estimate.is_complete
    assert {item.category for item in estimate.line_items} == {
        "realtime.input.text",
        "realtime.output.text",
    }


@pytest.mark.parametrize("audio_cache_read", [5e-6, None])
def test_maps_tokenary_realtime_fields(
    monkeypatch: pytest.MonkeyPatch,
    audio_cache_read: float | None,
) -> None:
    pricing = GeneratedModelPricing(
        input_cost_per_token=1e-6,
        cache_read_input_token_cost=2e-6,
        output_cost_per_token=3e-6,
        input_cost_per_audio_token=4e-6,
        cache_read_input_audio_token_cost=audio_cache_read,
        cache_creation_input_audio_token_cost=6e-6,
        output_cost_per_audio_token=7e-6,
        input_cost_per_image=8e-6,
        cache_read_input_image_token_cost=9e-6,
    )
    monkeypatch.setitem(
        pricing_module.MODEL_PRICINGS_BY_NAME,
        ModelName.GPT_REALTIME,
        pricing,
    )

    rates = pricing_module._realtime_rates(ModelName.GPT_REALTIME)

    assert rates == RealtimeRates(
        text_input=Decimal(1),
        text_cached_input=Decimal(2),
        text_output=Decimal(3),
        audio_input=Decimal(4),
        # cache_creation_..., not cache_read_..., is litellm's actual field
        # for the realtime audio-cache discount; cache_read_... is ignored.
        audio_cached_input=Decimal(6),
        audio_output=Decimal(7),
        image_input=Decimal(8),
        image_cached_input=Decimal(9),
    )


@pytest.mark.asyncio
async def test_marks_missing_tokenary_rate_incomplete() -> None:
    event_bus = EventBus()
    tracker = TokenTracker(
        event_bus=event_bus,
        realtime_model="gpt-realtime-mini",
    )

    await event_bus.dispatch(response_done())

    report = tracker.report()
    assert not report.cost.is_complete
    assert "No price configured for realtime.input.text.cached." in report.cost.notes
    assert "No price configured for realtime.input.image." in report.cost.notes


@pytest.mark.asyncio
async def test_unknown_realtime_model_is_incomplete() -> None:
    event_bus = EventBus()
    tracker = TokenTracker(
        event_bus=event_bus,
        realtime_model="unknown-realtime-model",
    )

    await event_bus.dispatch(response_done())

    report = tracker.report()
    assert report.cost.total == 0
    assert not report.cost.is_complete
    assert report.cost.notes == [
        "No realtime pricing configured for unknown-realtime-model."
    ]


@pytest.mark.asyncio
async def test_custom_catalog_overrides_tokenary_rates() -> None:
    event_bus = EventBus()
    custom_rate = Decimal(1)
    catalog = PricingCatalog(
        realtime={
            "custom-realtime": RealtimeRates(
                text_input=custom_rate,
                text_cached_input=custom_rate,
                text_output=custom_rate,
                audio_input=custom_rate,
                audio_cached_input=custom_rate,
                audio_output=custom_rate,
                image_input=custom_rate,
                image_cached_input=custom_rate,
            )
        },
        transcription={},
    )
    tracker = TokenTracker(
        event_bus=event_bus,
        realtime_model="custom-realtime",
        pricing_catalog=catalog,
    )

    await event_bus.dispatch(response_done())

    report = tracker.report()
    assert report.cost.total == Decimal("0.000205")
    assert report.cost.is_complete


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
