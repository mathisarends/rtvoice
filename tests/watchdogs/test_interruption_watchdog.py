from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.events.views import AssistantInterruptedEvent
from rtvoice.handler import InterruptionHandler
from rtvoice.realtime.schemas import (
    ConversationItemTruncateEvent,
    InputAudioBufferSpeechStartedEvent,
    RealtimeResponseObject,
    RealtimeServerEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputAudioDeltaEvent,
)


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def websocket() -> AsyncMock:
    ws = AsyncMock()
    ws.send = AsyncMock()
    return ws


@pytest.fixture
def audio_session() -> MagicMock:
    session = MagicMock()
    session.is_playing = False
    session.clear_output_buffer = AsyncMock()
    return session


@pytest.fixture
def watchdog(
    event_bus: EventBus, websocket: AsyncMock, audio_session: MagicMock
) -> InterruptionHandler:
    return InterruptionHandler(event_bus, websocket, audio_session)


def make_response_created(response_id: str = "resp_001") -> ResponseCreatedEvent:
    return ResponseCreatedEvent(
        type=RealtimeServerEvent.RESPONSE_CREATED,
        event_id="evt_001",
        response=RealtimeResponseObject(id=response_id),
    )


def make_response_done(response_id: str = "resp_001") -> ResponseDoneEvent:
    return ResponseDoneEvent(
        type=RealtimeServerEvent.RESPONSE_DONE,
        event_id="evt_002",
        response=RealtimeResponseObject(id=response_id),
    )


def make_audio_delta(
    response_id: str = "resp_001", item_id: str = "item_001"
) -> ResponseOutputAudioDeltaEvent:
    return ResponseOutputAudioDeltaEvent(
        event_id="evt_003",
        item_id=item_id,
        response_id=response_id,
        output_index=0,
        content_index=0,
        delta="AAAA",
    )


def make_speech_started() -> InputAudioBufferSpeechStartedEvent:
    return InputAudioBufferSpeechStartedEvent(
        event_id="evt_004",
        item_id="item_002",
        audio_start_ms=500,
    )


class TestStateTracking:
    @pytest.mark.asyncio
    async def test_response_created_sets_response_id(
        self, event_bus: EventBus, watchdog: InterruptionHandler
    ) -> None:
        await event_bus.dispatch(make_response_created("resp_abc"))

        assert watchdog._response_id == "resp_abc"

    @pytest.mark.asyncio
    async def test_response_created_sets_assistant_speaking(
        self, event_bus: EventBus, watchdog: InterruptionHandler
    ) -> None:
        await event_bus.dispatch(make_response_created())

        assert watchdog._assistant_is_speaking is True

    @pytest.mark.asyncio
    async def test_audio_delta_tracks_item_id_for_matching_response(
        self, event_bus: EventBus, watchdog: InterruptionHandler
    ) -> None:
        await event_bus.dispatch(make_response_created("resp_001"))
        await event_bus.dispatch(
            make_audio_delta(response_id="resp_001", item_id="item_xyz")
        )

        assert watchdog._item_id == "item_xyz"

    @pytest.mark.asyncio
    async def test_audio_delta_does_not_track_item_id_for_different_response(
        self, event_bus: EventBus, watchdog: InterruptionHandler
    ) -> None:
        await event_bus.dispatch(make_response_created("resp_001"))
        await event_bus.dispatch(
            make_audio_delta(response_id="resp_other", item_id="item_xyz")
        )

        assert watchdog._item_id is None

    @pytest.mark.asyncio
    async def test_response_done_resets_state(
        self, event_bus: EventBus, watchdog: InterruptionHandler
    ) -> None:
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(make_audio_delta())
        await event_bus.dispatch(make_response_done())

        assert watchdog._response_id is None
        assert watchdog._item_id is None
        assert watchdog._assistant_is_speaking is False

    @pytest.mark.asyncio
    async def test_response_done_for_different_response_does_not_reset(
        self, event_bus: EventBus, watchdog: InterruptionHandler
    ) -> None:
        await event_bus.dispatch(make_response_created("resp_001"))
        await event_bus.dispatch(make_response_done("resp_other"))

        assert watchdog._response_id == "resp_001"
        assert watchdog._assistant_is_speaking is True


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_barge_in_sends_response_cancel_when_assistant_speaking(
        self,
        event_bus: EventBus,
        watchdog: InterruptionHandler,
        websocket: AsyncMock,
    ) -> None:
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(make_speech_started())

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ResponseCancelEvent in sent_types

    @pytest.mark.asyncio
    async def test_barge_in_sends_truncate_when_item_tracked(
        self,
        event_bus: EventBus,
        watchdog: InterruptionHandler,
        websocket: AsyncMock,
    ) -> None:
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(make_audio_delta())
        await event_bus.dispatch(make_speech_started())

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ConversationItemTruncateEvent in sent_types

    @pytest.mark.asyncio
    async def test_barge_in_dispatches_assistant_interrupted_event(
        self,
        event_bus: EventBus,
        watchdog: InterruptionHandler,
    ) -> None:
        received: list[AssistantInterruptedEvent] = []

        async def capture(e: AssistantInterruptedEvent) -> None:
            received.append(e)

        event_bus.subscribe(AssistantInterruptedEvent, capture)
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(make_speech_started())

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_barge_in_resets_state(
        self, event_bus: EventBus, watchdog: InterruptionHandler
    ) -> None:
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(make_audio_delta())
        await event_bus.dispatch(make_speech_started())

        assert watchdog._response_id is None
        assert watchdog._item_id is None
        assert watchdog._assistant_is_speaking is False

    @pytest.mark.asyncio
    async def test_no_barge_in_when_assistant_not_speaking_and_not_playing(
        self,
        event_bus: EventBus,
        watchdog: InterruptionHandler,
        websocket: AsyncMock,
        audio_session: MagicMock,
    ) -> None:
        audio_session.is_playing = False

        await event_bus.dispatch(make_speech_started())

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_barge_in_dispatches_interrupted_event_when_audio_still_playing(
        self,
        event_bus: EventBus,
        watchdog: InterruptionHandler,
        audio_session: MagicMock,
    ) -> None:
        received: list[AssistantInterruptedEvent] = []

        async def capture(e: AssistantInterruptedEvent) -> None:
            received.append(e)

        event_bus.subscribe(AssistantInterruptedEvent, capture)
        audio_session.is_playing = True

        await event_bus.dispatch(make_speech_started())

        assert len(received) == 1
