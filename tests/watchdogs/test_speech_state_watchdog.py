import pytest

from rtvoice.events.bus import EventBus
from rtvoice.events.views import (
    AssistantStartedRespondingEvent,
    AssistantStoppedRespondingEvent,
    AudioPlaybackCompletedEvent,
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeResponseObject,
    RealtimeServerEvent,
    ResponseCreatedEvent,
)
from rtvoice.watchdogs import SpeechStateWatchdog


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def watchdog(event_bus: EventBus) -> SpeechStateWatchdog:
    return SpeechStateWatchdog(event_bus)


def make_speech_started() -> InputAudioBufferSpeechStartedEvent:
    return InputAudioBufferSpeechStartedEvent(
        event_id="evt_001",
        item_id="item_001",
        audio_start_ms=0,
    )


def make_speech_stopped() -> InputAudioBufferSpeechStoppedEvent:
    return InputAudioBufferSpeechStoppedEvent(
        event_id="evt_002",
        item_id="item_001",
        audio_end_ms=1500,
    )


def make_response_created(response_id: str = "resp_001") -> ResponseCreatedEvent:
    return ResponseCreatedEvent(
        type=RealtimeServerEvent.RESPONSE_CREATED,
        event_id="evt_003",
        response=RealtimeResponseObject(id=response_id),
    )


class TestUserSpeech:
    @pytest.mark.asyncio
    async def test_speech_started_dispatches_user_started_speaking(
        self, event_bus: EventBus, watchdog: SpeechStateWatchdog
    ) -> None:
        received: list[UserStartedSpeakingEvent] = []

        async def capture(e: UserStartedSpeakingEvent) -> None:
            received.append(e)

        event_bus.subscribe(UserStartedSpeakingEvent, capture)
        await event_bus.dispatch(make_speech_started())

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_speech_stopped_dispatches_user_stopped_speaking(
        self, event_bus: EventBus, watchdog: SpeechStateWatchdog
    ) -> None:
        received: list[UserStoppedSpeakingEvent] = []

        async def capture(e: UserStoppedSpeakingEvent) -> None:
            received.append(e)

        event_bus.subscribe(UserStoppedSpeakingEvent, capture)
        await event_bus.dispatch(make_speech_stopped())

        assert len(received) == 1


class TestAssistantSpeech:
    @pytest.mark.asyncio
    async def test_response_created_dispatches_assistant_started_responding(
        self, event_bus: EventBus, watchdog: SpeechStateWatchdog
    ) -> None:
        received: list[AssistantStartedRespondingEvent] = []

        async def capture(e: AssistantStartedRespondingEvent) -> None:
            received.append(e)

        event_bus.subscribe(AssistantStartedRespondingEvent, capture)
        await event_bus.dispatch(make_response_created())

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_playback_completed_dispatches_assistant_stopped_responding(
        self, event_bus: EventBus, watchdog: SpeechStateWatchdog
    ) -> None:
        received: list[AssistantStoppedRespondingEvent] = []

        async def capture(e: AssistantStoppedRespondingEvent) -> None:
            received.append(e)

        event_bus.subscribe(AssistantStoppedRespondingEvent, capture)
        await event_bus.dispatch(AudioPlaybackCompletedEvent())

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_multiple_responses_dispatch_multiple_started_events(
        self, event_bus: EventBus, watchdog: SpeechStateWatchdog
    ) -> None:
        received: list[AssistantStartedRespondingEvent] = []

        async def capture(e: AssistantStartedRespondingEvent) -> None:
            received.append(e)

        event_bus.subscribe(AssistantStartedRespondingEvent, capture)
        await event_bus.dispatch(make_response_created("resp_001"))
        await event_bus.dispatch(make_response_created("resp_002"))

        assert len(received) == 2
