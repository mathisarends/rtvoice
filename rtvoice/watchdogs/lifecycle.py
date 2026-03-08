import logging

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentSessionConnectedEvent,
    AgentStoppedEvent,
    StartAgentCommand,
    UpdateSpeechSpeedCommand,
)
from rtvoice.realtime.schemas import (
    AudioConfig,
    AudioInputConfig,
    AudioOutputConfig,
    InputAudioBufferAppendEvent,
    InputAudioNoiseReductionConfig,
    InputAudioTranscriptionConfig,
    NoiseReductionType,
    RealtimeSessionConfig,
    SemanticVADConfig,
    ServerVADConfig,
    SessionUpdateEvent,
    SpeedUpdateEvent,
    ToolChoiceMode,
    TurnDetectionConfig,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.shared.decorators import timed
from rtvoice.views import SemanticVAD, ServerVAD, TurnDetection

logger = logging.getLogger(__name__)


class LifecycleWatchdog:
    def __init__(self, event_bus: EventBus, websocket: RealtimeWebSocket):
        self._event_bus = event_bus
        self._websocket = websocket
        self._subscribe(event_bus)

    def _subscribe(self, event_bus: EventBus) -> None:
        event_bus.subscribe(StartAgentCommand, self._on_start_agent_command)
        event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)
        event_bus.subscribe(
            InputAudioBufferAppendEvent, self._on_input_audio_buffer_append
        )
        event_bus.subscribe(UpdateSpeechSpeedCommand, self._on_update_speech_speed)

    @timed()
    async def _on_start_agent_command(self, command: StartAgentCommand) -> None:
        logger.info("Starting agent session")

        if not self._websocket.is_connected:
            await self._websocket.connect()

        session_config = self._build_session_config(command)
        await self._websocket.send(SessionUpdateEvent(session=session_config))
        await self._event_bus.dispatch(AgentSessionConnectedEvent())

        logger.info("Agent session ready")

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        if not self._websocket.is_connected:
            return
        await self._websocket.close()
        logger.info("Agent session stopped")

    async def _on_input_audio_buffer_append(
        self, event: InputAudioBufferAppendEvent
    ) -> None:
        if not self._websocket.is_connected:
            logger.warning("Cannot send audio — WebSocket not connected")
            return
        await self._websocket.send(event)

    async def _on_update_speech_speed(self, command: UpdateSpeechSpeedCommand) -> None:
        if not self._websocket.is_connected:
            logger.warning("Cannot update speed — WebSocket not connected")
            return
        await self._websocket.send(SpeedUpdateEvent.from_speed(command.speed))

    def _build_session_config(
        self, command: StartAgentCommand
    ) -> RealtimeSessionConfig:
        return RealtimeSessionConfig(
            model=command.model,
            instructions=command.instructions,
            tool_choice=ToolChoiceMode.AUTO,
            tools=command.tools.get_tool_schema(),
            audio=AudioConfig(
                input=self._build_audio_input_config(command),
                output=AudioOutputConfig(
                    voice=command.voice.value,
                    speed=command.speech_speed,
                ),
            ),
        )

    def _build_audio_input_config(self, command: StartAgentCommand) -> AudioInputConfig:
        return AudioInputConfig(
            turn_detection=self._build_turn_detection_config(command.turn_detection),
            noise_reduction=InputAudioNoiseReductionConfig(
                type=NoiseReductionType(command.noise_reduction),
            ),
            transcription=InputAudioTranscriptionConfig(
                model=command.transcription_model
            )
            if command.transcription_model is not None
            else None,
        )

    def _build_turn_detection_config(
        self, turn_detection: TurnDetection
    ) -> TurnDetectionConfig:
        if isinstance(turn_detection, SemanticVAD):
            return SemanticVADConfig(eagerness=turn_detection.eagerness)

        if isinstance(turn_detection, ServerVAD):
            return ServerVADConfig(
                threshold=turn_detection.threshold,
                prefix_padding_ms=turn_detection.prefix_padding_ms,
                silence_duration_ms=turn_detection.silence_duration_ms,
            )

        raise TypeError(f"Unknown TurnDetection type: {type(turn_detection)}")
