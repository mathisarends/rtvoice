from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.handler import AudioHandler
from rtvoice.realtime.schemas import InputAudioBufferAppendEvent


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def websocket() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.is_connected = False
    return ws


@pytest.fixture
def audio_session() -> MagicMock:
    return MagicMock()


@pytest.fixture
def audio_handler(
    event_bus: EventBus, audio_session: MagicMock, websocket: MagicMock
) -> AudioHandler:
    return AudioHandler(event_bus, audio_session, websocket)


class TestAudioForwarding:
    @pytest.mark.asyncio
    async def test_forwards_audio_event_when_connected(
        self,
        event_bus: EventBus,
        audio_handler: AudioHandler,
        websocket: MagicMock,
    ) -> None:
        websocket.is_connected = True
        audio_event = InputAudioBufferAppendEvent(audio="AAAA")

        await event_bus.dispatch(audio_event)

        websocket.send.assert_called_once_with(audio_event)

    @pytest.mark.asyncio
    async def test_does_not_forward_when_disconnected(
        self,
        event_bus: EventBus,
        audio_handler: AudioHandler,
        websocket: MagicMock,
    ) -> None:
        websocket.is_connected = False

        await event_bus.dispatch(InputAudioBufferAppendEvent(audio="AAAA"))

        websocket.send.assert_not_called()
