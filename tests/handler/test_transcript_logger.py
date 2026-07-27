import logging

import pytest
from transitbus import EventBus

from rtvoice.events.views import (
    AssistantTranscriptCompletedEvent,
    AssistantTranscriptDeltaEvent,
    UserTranscriptCompletedEvent,
)
from rtvoice.handler import TranscriptLogger

LOGGER = "rtvoice.handler.transcript_logger"


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def transcript_logger(event_bus: EventBus) -> TranscriptLogger:
    return TranscriptLogger(event_bus)


@pytest.mark.asyncio
async def test_logs_completed_user_transcript(
    event_bus: EventBus,
    transcript_logger: TranscriptLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await event_bus.dispatch(
            UserTranscriptCompletedEvent(
                transcript="Das habe ich gesagt.", item_id="item-1"
            )
        )

    assert caplog.messages == ["[user] Das habe ich gesagt."]


@pytest.mark.asyncio
async def test_logs_completed_assistant_transcript(
    event_bus: EventBus,
    transcript_logger: TranscriptLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await event_bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="Das hat der Agent gesagt.",
                item_id="item-2",
                output_index=0,
                content_index=0,
                response_id="response-1",
            )
        )

    assert caplog.messages == ["[assistant] Das hat der Agent gesagt."]


@pytest.mark.asyncio
async def test_does_not_log_partial_transcripts(
    event_bus: EventBus,
    transcript_logger: TranscriptLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await event_bus.dispatch(
            AssistantTranscriptDeltaEvent(
                delta="Noch nicht vollständig",
                item_id="item-2",
                output_index=0,
                content_index=0,
            )
        )

    assert caplog.messages == []
