import base64
from unittest.mock import AsyncMock, MagicMock

import pytest
from transitbus import EventBus

from rtvoice.events.views import (
    AssistantInterruptedEvent,
    AudioPlaybackCompletedEvent,
    InterruptAssistantCommand,
)
from rtvoice.handler import BargeInCoordinator
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
def coordinator(
    event_bus: EventBus, websocket: AsyncMock, audio_session: MagicMock
) -> BargeInCoordinator:
    return BargeInCoordinator(event_bus, websocket, audio_session)


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
        delta=base64.b64encode(bytes(96_000)).decode(),
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
        self, event_bus: EventBus, coordinator: BargeInCoordinator
    ) -> None:
        await event_bus.dispatch(make_response_created("resp_abc"))

        assert coordinator._response_id == "resp_abc"

    @pytest.mark.asyncio
    async def test_response_created_sets_assistant_speaking(
        self, event_bus: EventBus, coordinator: BargeInCoordinator
    ) -> None:
        await event_bus.dispatch(make_response_created())

        assert coordinator._assistant_is_speaking is True

    @pytest.mark.asyncio
    async def test_response_created_discards_previous_playback_tracking(
        self, event_bus: EventBus, coordinator: BargeInCoordinator
    ) -> None:
        await event_bus.dispatch(make_response_created("resp_old"))
        await event_bus.dispatch(make_audio_delta(response_id="resp_old"))

        await event_bus.dispatch(make_response_created("resp_new"))

        assert coordinator._item_id is None
        assert coordinator._start_time is None
        assert coordinator._audio_bytes == 0

    @pytest.mark.asyncio
    async def test_playback_timer_starts_with_first_audio_chunk(
        self,
        event_bus: EventBus,
        coordinator: BargeInCoordinator,
    ) -> None:
        coordinator._clock = MagicMock(return_value=123.0)

        await event_bus.dispatch(make_response_created())
        assert coordinator._start_time is None

        await event_bus.dispatch(make_audio_delta())
        assert coordinator._start_time == 123.0

    @pytest.mark.asyncio
    async def test_audio_delta_tracks_item_id_for_matching_response(
        self, event_bus: EventBus, coordinator: BargeInCoordinator
    ) -> None:
        await event_bus.dispatch(make_response_created("resp_001"))
        await event_bus.dispatch(
            make_audio_delta(response_id="resp_001", item_id="item_xyz")
        )

        assert coordinator._item_id == "item_xyz"

    @pytest.mark.asyncio
    async def test_audio_delta_does_not_track_item_id_for_different_response(
        self, event_bus: EventBus, coordinator: BargeInCoordinator
    ) -> None:
        await event_bus.dispatch(make_response_created("resp_001"))
        await event_bus.dispatch(
            make_audio_delta(response_id="resp_other", item_id="item_xyz")
        )

        assert coordinator._item_id is None

    @pytest.mark.asyncio
    async def test_response_done_retains_state_until_playback_completes(
        self, event_bus: EventBus, coordinator: BargeInCoordinator
    ) -> None:
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(make_audio_delta())
        await event_bus.dispatch(make_response_done())

        assert coordinator._response_id == "resp_001"
        assert coordinator._item_id == "item_001"
        assert coordinator._assistant_is_speaking is False

        await event_bus.dispatch(AudioPlaybackCompletedEvent())

        assert coordinator._response_id is None
        assert coordinator._item_id is None
        assert coordinator._assistant_is_speaking is False

    @pytest.mark.asyncio
    async def test_response_done_for_different_response_does_not_reset(
        self, event_bus: EventBus, coordinator: BargeInCoordinator
    ) -> None:
        await event_bus.dispatch(make_response_created("resp_001"))
        await event_bus.dispatch(make_response_done("resp_other"))

        assert coordinator._response_id == "resp_001"
        assert coordinator._assistant_is_speaking is True


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_barge_in_sends_response_cancel_when_assistant_speaking(
        self,
        event_bus: EventBus,
        coordinator: BargeInCoordinator,
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
        coordinator: BargeInCoordinator,
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
        coordinator: BargeInCoordinator,
    ) -> None:
        received: list[AssistantInterruptedEvent] = []

        async def capture(e: AssistantInterruptedEvent) -> None:
            received.append(e)

        event_bus.on(AssistantInterruptedEvent, capture)
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(make_speech_started())

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_barge_in_resets_state(
        self, event_bus: EventBus, coordinator: BargeInCoordinator
    ) -> None:
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(make_audio_delta())
        await event_bus.dispatch(make_speech_started())

        assert coordinator._response_id is None
        assert coordinator._item_id is None
        assert coordinator._assistant_is_speaking is False

    @pytest.mark.asyncio
    async def test_no_barge_in_when_assistant_not_speaking_and_not_playing(
        self,
        event_bus: EventBus,
        coordinator: BargeInCoordinator,
        websocket: AsyncMock,
        audio_session: MagicMock,
    ) -> None:
        audio_session.is_playing = False

        await event_bus.dispatch(make_speech_started())

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_interrupt_command_cancels_running_response(
        self,
        event_bus: EventBus,
        coordinator: BargeInCoordinator,
        websocket: AsyncMock,
    ) -> None:
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(InterruptAssistantCommand())

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ResponseCancelEvent in sent_types

    @pytest.mark.asyncio
    async def test_interrupt_command_dispatches_assistant_interrupted_event(
        self,
        event_bus: EventBus,
        coordinator: BargeInCoordinator,
    ) -> None:
        received: list[AssistantInterruptedEvent] = []
        coordinator._clock = MagicMock(side_effect=[100.0, 102.0])

        async def capture(e: AssistantInterruptedEvent) -> None:
            received.append(e)

        event_bus.on(AssistantInterruptedEvent, capture)
        coordinator.set_speech_speed(1.5)
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(make_audio_delta())
        await event_bus.dispatch(InterruptAssistantCommand())

        assert len(received) == 1
        assert received[0].response_id == "resp_001"
        assert received[0].played_ms == 2_000
        assert received[0].speech_speed == 1.5

    @pytest.mark.asyncio
    async def test_interrupt_command_is_noop_when_nothing_is_playing(
        self,
        event_bus: EventBus,
        coordinator: BargeInCoordinator,
        websocket: AsyncMock,
    ) -> None:
        await event_bus.dispatch(InterruptAssistantCommand())

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_barge_in_dispatches_interrupted_event_when_audio_still_playing(
        self,
        event_bus: EventBus,
        coordinator: BargeInCoordinator,
        audio_session: MagicMock,
    ) -> None:
        received: list[AssistantInterruptedEvent] = []

        async def capture(e: AssistantInterruptedEvent) -> None:
            received.append(e)

        event_bus.on(AssistantInterruptedEvent, capture)
        audio_session.is_playing = True

        await event_bus.dispatch(make_speech_started())

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_barge_in_after_playback_ended_ignores_stale_item(
        self,
        event_bus: EventBus,
        coordinator: BargeInCoordinator,
        audio_session: MagicMock,
        websocket: AsyncMock,
    ) -> None:
        received: list[AssistantInterruptedEvent] = []

        async def capture(e: AssistantInterruptedEvent) -> None:
            received.append(e)

        event_bus.on(AssistantInterruptedEvent, capture)
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(make_audio_delta())
        await event_bus.dispatch(make_response_done())
        audio_session.is_playing = False

        await event_bus.dispatch(make_speech_started())

        assert received == []
        websocket.send.assert_not_awaited()
        assert coordinator._item_id is None

    @pytest.mark.asyncio
    async def test_truncate_time_does_not_exceed_received_audio(
        self,
        event_bus: EventBus,
        coordinator: BargeInCoordinator,
        audio_session: MagicMock,
        websocket: AsyncMock,
    ) -> None:
        coordinator._clock = MagicMock(side_effect=[100.0, 114.2])
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(make_audio_delta())
        await event_bus.dispatch(make_response_done())
        audio_session.is_playing = True

        await event_bus.dispatch(make_speech_started())

        truncate = next(
            call.args[0]
            for call in websocket.send.await_args_list
            if isinstance(call.args[0], ConversationItemTruncateEvent)
        )
        assert truncate.audio_end_ms == 2_000
