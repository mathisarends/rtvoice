from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.events.views import (
    AgentSessionConnectedEvent,
    AgentStoppedEvent,
    StartAgentCommand,
)
from rtvoice.realtime.schemas import InputAudioBufferAppendEvent, SessionUpdateEvent
from rtvoice.views import (
    AssistantVoice,
    NoiseReduction,
    RealtimeModel,
    SemanticVAD,
    TranscriptionModel,
)
from rtvoice.watchdogs import LifecycleWatchdog


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def websocket() -> MagicMock:
    ws = MagicMock()
    ws.connect = AsyncMock()
    ws.close = AsyncMock()
    ws.send = AsyncMock()
    ws.is_connected = False
    return ws


@pytest.fixture
def watchdog(event_bus: EventBus, websocket: MagicMock) -> LifecycleWatchdog:
    return LifecycleWatchdog(event_bus, websocket)


def make_start_command(**overrides) -> StartAgentCommand:
    tools = MagicMock()
    tools.get_tool_schema.return_value = []
    return StartAgentCommand(
        model=RealtimeModel.GPT_REALTIME_MINI,
        instructions="",
        voice=AssistantVoice.MARIN,
        speech_speed=1.0,
        transcription_model=TranscriptionModel.WHISPER_1,
        noise_reduction=NoiseReduction.FAR_FIELD,
        turn_detection=SemanticVAD(),
        tools=tools,
        **overrides,
    )


class TestStartAgent:
    @pytest.mark.asyncio
    async def test_connects_websocket_when_not_connected(
        self,
        event_bus: EventBus,
        watchdog: LifecycleWatchdog,
        websocket: MagicMock,
    ) -> None:
        websocket.is_connected = False

        await event_bus.dispatch(make_start_command())

        websocket.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_reconnect_when_already_connected(
        self,
        event_bus: EventBus,
        watchdog: LifecycleWatchdog,
        websocket: MagicMock,
    ) -> None:
        websocket.is_connected = True

        await event_bus.dispatch(make_start_command())

        websocket.connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_session_update_event(
        self,
        event_bus: EventBus,
        watchdog: LifecycleWatchdog,
        websocket: MagicMock,
    ) -> None:
        await event_bus.dispatch(make_start_command())

        websocket.send.assert_called_once()
        sent = websocket.send.call_args[0][0]
        assert isinstance(sent, SessionUpdateEvent)

    @pytest.mark.asyncio
    async def test_dispatches_agent_session_connected_event(
        self, event_bus: EventBus, watchdog: LifecycleWatchdog
    ) -> None:
        received: list[AgentSessionConnectedEvent] = []

        async def capture(e: AgentSessionConnectedEvent) -> None:
            received.append(e)

        event_bus.subscribe(AgentSessionConnectedEvent, capture)
        await event_bus.dispatch(make_start_command())

        assert len(received) == 1


class TestAgentStopped:
    @pytest.mark.asyncio
    async def test_closes_websocket_when_connected(
        self,
        event_bus: EventBus,
        watchdog: LifecycleWatchdog,
        websocket: MagicMock,
    ) -> None:
        websocket.is_connected = True

        await event_bus.dispatch(AgentStoppedEvent())

        websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_close_when_not_connected(
        self,
        event_bus: EventBus,
        watchdog: LifecycleWatchdog,
        websocket: MagicMock,
    ) -> None:
        websocket.is_connected = False

        await event_bus.dispatch(AgentStoppedEvent())

        websocket.close.assert_not_called()


class TestAudioForwarding:
    @pytest.mark.asyncio
    async def test_audio_event_forwarded_when_connected(
        self,
        event_bus: EventBus,
        watchdog: LifecycleWatchdog,
        websocket: MagicMock,
    ) -> None:
        websocket.is_connected = True
        audio_event = InputAudioBufferAppendEvent(audio="AAAA")

        await event_bus.dispatch(audio_event)

        websocket.send.assert_called_once_with(audio_event)

    @pytest.mark.asyncio
    async def test_audio_event_not_forwarded_when_disconnected(
        self,
        event_bus: EventBus,
        watchdog: LifecycleWatchdog,
        websocket: MagicMock,
    ) -> None:
        websocket.is_connected = False

        await event_bus.dispatch(InputAudioBufferAppendEvent(audio="AAAA"))

        websocket.send.assert_not_called()
