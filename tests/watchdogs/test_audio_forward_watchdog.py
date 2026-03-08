from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.realtime.schemas import InputAudioBufferAppendEvent
from rtvoice.watchdogs import AudioForwardWatchdog


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
def watchdog(event_bus: EventBus, websocket: MagicMock) -> AudioForwardWatchdog:
    return AudioForwardWatchdog(event_bus, websocket)


class TestAudioForwarding:
    @pytest.mark.asyncio
    async def test_forwards_audio_event_when_connected(
        self,
        event_bus: EventBus,
        watchdog: AudioForwardWatchdog,
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
        watchdog: AudioForwardWatchdog,
        websocket: MagicMock,
    ) -> None:
        websocket.is_connected = False

        await event_bus.dispatch(InputAudioBufferAppendEvent(audio="AAAA"))

        websocket.send.assert_not_called()
