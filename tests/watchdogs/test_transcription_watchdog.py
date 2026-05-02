import pytest

from rtvoice.events.bus import EventBus
from rtvoice.events.views import (
    AssistantTranscriptCompletedEvent,
    AssistantTranscriptDeltaEvent,
    UserTranscriptCompletedEvent,
)
from rtvoice.handler import TranscriptionAccumulator
from rtvoice.realtime.schemas import (
    InputAudioTranscriptionCompleted,
    RealtimeServerEvent,
    ResponseOutputAudioTranscriptDone,
    ResponseOutputTextDelta,
    ResponseOutputTextDone,
    ResponseTextDelta,
)


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def watchdog(event_bus: EventBus) -> TranscriptionAccumulator:
    return TranscriptionAccumulator(event_bus)


class TestUserTranscription:
    @pytest.mark.asyncio
    async def test_transcription_completed_dispatches_user_transcript_event(
        self, event_bus: EventBus, watchdog: TranscriptionAccumulator
    ) -> None:
        received: list[UserTranscriptCompletedEvent] = []

        async def capture(e: UserTranscriptCompletedEvent) -> None:
            received.append(e)

        event_bus.subscribe(UserTranscriptCompletedEvent, capture)
        await event_bus.dispatch(
            InputAudioTranscriptionCompleted(
                type=RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED,
                event_id="evt_001",
                item_id="item_001",
                content_index=0,
                transcript="Hello, world!",
            )
        )

        assert len(received) == 1
        assert received[0].transcript == "Hello, world!"
        assert received[0].item_id == "item_001"

    @pytest.mark.asyncio
    async def test_user_transcript_carries_item_id(
        self, event_bus: EventBus, watchdog: TranscriptionAccumulator
    ) -> None:
        received: list[UserTranscriptCompletedEvent] = []

        async def capture(e: UserTranscriptCompletedEvent) -> None:
            received.append(e)

        event_bus.subscribe(UserTranscriptCompletedEvent, capture)
        await event_bus.dispatch(
            InputAudioTranscriptionCompleted(
                type=RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED,
                event_id="evt_002",
                item_id="item_xyz",
                content_index=0,
                transcript="Some text",
            )
        )

        assert received[0].item_id == "item_xyz"


class TestAssistantTranscription:
    @pytest.mark.asyncio
    async def test_transcript_done_dispatches_assistant_transcript_event(
        self, event_bus: EventBus, watchdog: TranscriptionAccumulator
    ) -> None:
        received: list[AssistantTranscriptCompletedEvent] = []

        async def capture(e: AssistantTranscriptCompletedEvent) -> None:
            received.append(e)

        event_bus.subscribe(AssistantTranscriptCompletedEvent, capture)
        await event_bus.dispatch(
            ResponseOutputAudioTranscriptDone(
                type=RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE,
                event_id="evt_003",
                item_id="item_002",
                response_id="resp_001",
                output_index=0,
                content_index=0,
                transcript="I am the assistant.",
            )
        )

        assert len(received) == 1
        assert received[0].transcript == "I am the assistant."
        assert received[0].item_id == "item_002"
        assert received[0].output_index == 0
        assert received[0].content_index == 0

    @pytest.mark.asyncio
    async def test_assistant_transcript_carries_output_and_content_index(
        self, event_bus: EventBus, watchdog: TranscriptionAccumulator
    ) -> None:
        received: list[AssistantTranscriptCompletedEvent] = []

        async def capture(e: AssistantTranscriptCompletedEvent) -> None:
            received.append(e)

        event_bus.subscribe(AssistantTranscriptCompletedEvent, capture)
        await event_bus.dispatch(
            ResponseOutputAudioTranscriptDone(
                type=RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE,
                event_id="evt_004",
                item_id="item_003",
                response_id="resp_002",
                output_index=2,
                content_index=1,
                transcript="Second response.",
            )
        )

        assert received[0].output_index == 2
        assert received[0].content_index == 1

    @pytest.mark.asyncio
    async def test_text_delta_dispatches_assistant_delta_event(
        self, event_bus: EventBus, watchdog: TranscriptionAccumulator
    ) -> None:
        received: list[AssistantTranscriptDeltaEvent] = []

        async def capture(e: AssistantTranscriptDeltaEvent) -> None:
            received.append(e)

        event_bus.subscribe(AssistantTranscriptDeltaEvent, capture)
        await event_bus.dispatch(
            ResponseOutputTextDelta(
                type=RealtimeServerEvent.RESPONSE_OUTPUT_TEXT_DELTA,
                event_id="evt_005",
                item_id="item_004",
                response_id="resp_003",
                output_index=0,
                content_index=0,
                delta="Hello ",
            )
        )

        assert len(received) == 1
        assert received[0].delta == "Hello "
        assert received[0].item_id == "item_004"

    @pytest.mark.asyncio
    async def test_text_done_dispatches_assistant_completed_event(
        self, event_bus: EventBus, watchdog: TranscriptionAccumulator
    ) -> None:
        received: list[AssistantTranscriptCompletedEvent] = []

        async def capture(e: AssistantTranscriptCompletedEvent) -> None:
            received.append(e)

        event_bus.subscribe(AssistantTranscriptCompletedEvent, capture)
        await event_bus.dispatch(
            ResponseOutputTextDone(
                type=RealtimeServerEvent.RESPONSE_OUTPUT_TEXT_DONE,
                event_id="evt_006",
                item_id="item_005",
                response_id="resp_004",
                output_index=1,
                content_index=0,
                text="Completed text",
            )
        )

        assert len(received) == 1
        assert received[0].transcript == "Completed text"

    @pytest.mark.asyncio
    async def test_response_text_delta_dispatches_assistant_delta_event(
        self, event_bus: EventBus, watchdog: TranscriptionAccumulator
    ) -> None:
        received: list[AssistantTranscriptDeltaEvent] = []

        async def capture(e: AssistantTranscriptDeltaEvent) -> None:
            received.append(e)

        event_bus.subscribe(AssistantTranscriptDeltaEvent, capture)
        await event_bus.dispatch(
            ResponseTextDelta(
                type=RealtimeServerEvent.RESPONSE_TEXT_DELTA,
                event_id="evt_007",
                item_id="item_006",
                response_id="resp_005",
                output_index=0,
                content_index=0,
                delta="world",
            )
        )

        assert len(received) == 1
        assert received[0].delta == "world"
