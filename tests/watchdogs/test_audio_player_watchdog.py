import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.events.views import (
    AgentSessionConnectedEvent,
    AgentStoppedEvent,
    AudioPlaybackCompletedEvent,
)
from rtvoice.handler import AudioPlayer
from rtvoice.realtime.schemas import (
    InputAudioBufferSpeechStartedEvent,
    RealtimeResponseObject,
    RealtimeServerEvent,
    ResponseDoneEvent,
    ResponseOutputAudioDeltaEvent,
)


async def _empty_stream():
    return
    yield


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def audio_session() -> MagicMock:
    session = MagicMock()
    session.start = AsyncMock()
    session.stop = AsyncMock()
    session.play_chunk = AsyncMock()
    session.clear_output_buffer = AsyncMock()
    session.stream_input_chunks = MagicMock(return_value=_empty_stream())
    session.is_playing = False
    return session


@pytest.fixture
def player(event_bus: EventBus, audio_session: MagicMock) -> AudioPlayer:
    return AudioPlayer(event_bus, audio_session)


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_session_connected_starts_audio(
        self,
        event_bus: EventBus,
        player: AudioPlayer,
        audio_session: MagicMock,
    ) -> None:
        await event_bus.dispatch(AgentSessionConnectedEvent())
        await asyncio.sleep(0)

        audio_session.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_connected_creates_streaming_task(
        self, event_bus: EventBus, player: AudioPlayer
    ) -> None:
        assert player._streaming_task is None

        await event_bus.dispatch(AgentSessionConnectedEvent())

        assert player._streaming_task is not None

    @pytest.mark.asyncio
    async def test_agent_stopped_stops_audio(
        self,
        event_bus: EventBus,
        player: AudioPlayer,
        audio_session: MagicMock,
    ) -> None:
        await event_bus.dispatch(AgentSessionConnectedEvent())
        await asyncio.sleep(0)
        await event_bus.dispatch(AgentStoppedEvent())

        audio_session.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_stopped_clears_streaming_task(
        self, event_bus: EventBus, player: AudioPlayer
    ) -> None:
        await event_bus.dispatch(AgentSessionConnectedEvent())
        await asyncio.sleep(0)
        await event_bus.dispatch(AgentStoppedEvent())

        assert player._streaming_task is None


class TestAudioDelta:
    @pytest.mark.asyncio
    async def test_audio_delta_plays_decoded_chunk(
        self,
        event_bus: EventBus,
        player: AudioPlayer,
        audio_session: MagicMock,
    ) -> None:
        audio_bytes = b"\x00\x01\x02\x03"
        encoded = base64.b64encode(audio_bytes).decode()

        await event_bus.dispatch(
            ResponseOutputAudioDeltaEvent(
                event_id="evt_001",
                item_id="item_001",
                response_id="resp_001",
                output_index=0,
                content_index=0,
                delta=encoded,
            )
        )

        audio_session.play_chunk.assert_called_once_with(audio_bytes)


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_user_started_speaking_clears_output_buffer(
        self,
        event_bus: EventBus,
        player: AudioPlayer,
        audio_session: MagicMock,
    ) -> None:
        await event_bus.dispatch(
            InputAudioBufferSpeechStartedEvent(
                event_id="evt_001",
                item_id="item_001",
                audio_start_ms=0,
            )
        )

        audio_session.clear_output_buffer.assert_called_once()


class TestPlaybackCompletion:
    @pytest.mark.asyncio
    async def test_response_done_dispatches_playback_completed_when_not_playing(
        self,
        event_bus: EventBus,
        player: AudioPlayer,
        audio_session: MagicMock,
    ) -> None:
        received: list[AudioPlaybackCompletedEvent] = []

        async def capture(e: AudioPlaybackCompletedEvent) -> None:
            received.append(e)

        event_bus.subscribe(AudioPlaybackCompletedEvent, capture)
        audio_session.is_playing = False

        await event_bus.dispatch(
            ResponseDoneEvent(
                type=RealtimeServerEvent.RESPONSE_DONE,
                event_id="evt_001",
                response=RealtimeResponseObject(id="resp_001"),
            )
        )
        await asyncio.sleep(0)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_response_done_waits_while_playing_before_dispatching(
        self,
        event_bus: EventBus,
        player: AudioPlayer,
        audio_session: MagicMock,
    ) -> None:
        received: list[AudioPlaybackCompletedEvent] = []

        async def capture(e: AudioPlaybackCompletedEvent) -> None:
            received.append(e)

        event_bus.subscribe(AudioPlaybackCompletedEvent, capture)
        audio_session.is_playing = True

        await event_bus.dispatch(
            ResponseDoneEvent(
                type=RealtimeServerEvent.RESPONSE_DONE,
                event_id="evt_001",
                response=RealtimeResponseObject(id="resp_001"),
            )
        )
        await asyncio.sleep(0)

        assert len(received) == 0

        audio_session.is_playing = False
        await asyncio.sleep(0.1)

        assert len(received) == 1
