from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.events.views import ConfigureSessionCommand, UpdateSpeechSpeedCommand
from rtvoice.realtime.schemas import (
    SemanticVADConfig,
    ServerVADConfig,
    SessionUpdateEvent,
    SpeedUpdateEvent,
)
from rtvoice.views import (
    AssistantVoice,
    NoiseReduction,
    RealtimeModel,
    SemanticEagerness,
    SemanticVAD,
    ServerVAD,
    TranscriptionModel,
)
from rtvoice.watchdogs import SessionWatchdog


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
def watchdog(event_bus: EventBus, websocket: MagicMock) -> SessionWatchdog:
    return SessionWatchdog(event_bus, websocket)


def make_configure_command(**overrides) -> ConfigureSessionCommand:
    tools = MagicMock()
    tools.get_tool_schema.return_value = []
    defaults: dict = dict(
        model=RealtimeModel.GPT_REALTIME_MINI,
        instructions="You are helpful.",
        voice=AssistantVoice.MARIN,
        speech_speed=1.0,
        transcription_model=TranscriptionModel.WHISPER_1,
        noise_reduction=NoiseReduction.FAR_FIELD,
        turn_detection=SemanticVAD(),
        tools=tools,
    )
    return ConfigureSessionCommand(**{**defaults, **overrides})


class TestConfigureSession:
    @pytest.mark.asyncio
    async def test_sends_session_update_event(
        self,
        event_bus: EventBus,
        watchdog: SessionWatchdog,
        websocket: MagicMock,
    ) -> None:
        await event_bus.dispatch(make_configure_command())

        websocket.send.assert_called_once()
        sent = websocket.send.call_args[0][0]
        assert isinstance(sent, SessionUpdateEvent)

    @pytest.mark.asyncio
    async def test_session_config_contains_model(
        self,
        event_bus: EventBus,
        watchdog: SessionWatchdog,
        websocket: MagicMock,
    ) -> None:
        await event_bus.dispatch(
            make_configure_command(model=RealtimeModel.GPT_REALTIME)
        )

        sent: SessionUpdateEvent = websocket.send.call_args[0][0]
        assert sent.session.model == RealtimeModel.GPT_REALTIME

    @pytest.mark.asyncio
    async def test_session_config_contains_instructions(
        self,
        event_bus: EventBus,
        watchdog: SessionWatchdog,
        websocket: MagicMock,
    ) -> None:
        await event_bus.dispatch(make_configure_command(instructions="Be concise."))

        sent: SessionUpdateEvent = websocket.send.call_args[0][0]
        assert sent.session.instructions == "Be concise."

    @pytest.mark.asyncio
    async def test_session_config_contains_voice(
        self,
        event_bus: EventBus,
        watchdog: SessionWatchdog,
        websocket: MagicMock,
    ) -> None:
        await event_bus.dispatch(make_configure_command(voice=AssistantVoice.CORAL))

        sent: SessionUpdateEvent = websocket.send.call_args[0][0]
        assert sent.session.audio.output.voice == AssistantVoice.CORAL.value

    @pytest.mark.asyncio
    async def test_session_config_contains_speech_speed(
        self,
        event_bus: EventBus,
        watchdog: SessionWatchdog,
        websocket: MagicMock,
    ) -> None:
        await event_bus.dispatch(make_configure_command(speech_speed=1.25))

        sent: SessionUpdateEvent = websocket.send.call_args[0][0]
        assert sent.session.audio.output.speed == 1.25

    @pytest.mark.asyncio
    async def test_semantic_vad_turn_detection(
        self,
        event_bus: EventBus,
        watchdog: SessionWatchdog,
        websocket: MagicMock,
    ) -> None:
        await event_bus.dispatch(
            make_configure_command(
                turn_detection=SemanticVAD(eagerness=SemanticEagerness.HIGH)
            )
        )

        sent: SessionUpdateEvent = websocket.send.call_args[0][0]
        turn = sent.session.audio.input.turn_detection
        assert isinstance(turn, SemanticVADConfig)
        assert turn.eagerness == SemanticEagerness.HIGH

    @pytest.mark.asyncio
    async def test_server_vad_turn_detection(
        self,
        event_bus: EventBus,
        watchdog: SessionWatchdog,
        websocket: MagicMock,
    ) -> None:
        await event_bus.dispatch(
            make_configure_command(
                turn_detection=ServerVAD(
                    threshold=0.7, prefix_padding_ms=200, silence_duration_ms=400
                )
            )
        )

        sent: SessionUpdateEvent = websocket.send.call_args[0][0]
        turn = sent.session.audio.input.turn_detection
        assert isinstance(turn, ServerVADConfig)
        assert turn.threshold == 0.7
        assert turn.prefix_padding_ms == 200
        assert turn.silence_duration_ms == 400

    @pytest.mark.asyncio
    async def test_transcription_model_included_when_set(
        self,
        event_bus: EventBus,
        watchdog: SessionWatchdog,
        websocket: MagicMock,
    ) -> None:
        await event_bus.dispatch(
            make_configure_command(transcription_model=TranscriptionModel.WHISPER_1)
        )

        sent: SessionUpdateEvent = websocket.send.call_args[0][0]
        assert sent.session.audio.input.transcription is not None
        assert (
            sent.session.audio.input.transcription.model == TranscriptionModel.WHISPER_1
        )

    @pytest.mark.asyncio
    async def test_transcription_is_none_when_not_set(
        self,
        event_bus: EventBus,
        watchdog: SessionWatchdog,
        websocket: MagicMock,
    ) -> None:
        await event_bus.dispatch(make_configure_command(transcription_model=None))

        sent: SessionUpdateEvent = websocket.send.call_args[0][0]
        assert sent.session.audio.input.transcription is None


class TestUpdateSpeechSpeed:
    @pytest.mark.asyncio
    async def test_sends_speed_update_when_connected(
        self,
        event_bus: EventBus,
        watchdog: SessionWatchdog,
        websocket: MagicMock,
    ) -> None:
        websocket.is_connected = True

        await event_bus.dispatch(UpdateSpeechSpeedCommand(speed=1.25))

        websocket.send.assert_called_once()
        sent = websocket.send.call_args[0][0]
        assert isinstance(sent, SpeedUpdateEvent)

    @pytest.mark.asyncio
    async def test_does_not_send_when_disconnected(
        self,
        event_bus: EventBus,
        watchdog: SessionWatchdog,
        websocket: MagicMock,
    ) -> None:
        websocket.is_connected = False

        await event_bus.dispatch(UpdateSpeechSpeedCommand(speed=1.25))

        websocket.send.assert_not_called()
