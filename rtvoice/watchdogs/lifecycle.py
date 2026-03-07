import logging

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentSessionConnectedEvent,
    AgentStoppedEvent,
    StartAgentCommand,
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
    ToolChoiceMode,
    TurnDetectionConfig,
)
from rtvoice.realtime.websocket.service import RealtimeWebSocket
from rtvoice.views import SemanticVAD, TurnDetection

logger = logging.getLogger(__name__)


class LifecycleWatchdog:
    def __init__(self, event_bus: EventBus, websocket: RealtimeWebSocket):
        self._event_bus = event_bus
        self._websocket = websocket

        self._event_bus.subscribe(StartAgentCommand, self._on_start_agent_command)
        self._event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)
        self._event_bus.subscribe(
            InputAudioBufferAppendEvent, self._on_input_audio_buffer_append
        )

    def _is_connected(self) -> bool:
        return self._websocket.is_connected

    async def _on_input_audio_buffer_append(
        self, event: InputAudioBufferAppendEvent
    ) -> None:
        if not self._is_connected():
            logger.warning("Cannot send audio - WebSocket not connected")
            return

        await self._websocket.send(event)

    async def _on_start_agent_command(self, command: StartAgentCommand) -> None:
        logger.info("Starting agent session")

        if not self._is_connected():
            await self._websocket.connect()

        session_config = self._build_session_config(command)
        await self._websocket.send(SessionUpdateEvent(session=session_config))
        await self._event_bus.dispatch(AgentSessionConnectedEvent())
        logger.info("Agent session ready")

    def _build_session_config(
        self, command: StartAgentCommand
    ) -> RealtimeSessionConfig:
        input_config = AudioInputConfig(
            transcription=InputAudioTranscriptionConfig(
                model=command.transcription_model
            ),
            noise_reduction=InputAudioNoiseReductionConfig(
                type=NoiseReductionType(command.noise_reduction)
            ),
            turn_detection=self._build_turn_detection_config(command.turn_detection),
        )
        return RealtimeSessionConfig(
            model=command.model,
            instructions=command.instructions,
            audio=AudioConfig(
                output=AudioOutputConfig(
                    speed=command.speech_speed,
                    voice=command.voice.value,
                ),
                input=input_config,
            ),
            tool_choice=ToolChoiceMode.AUTO,
            tools=command.tools.get_tool_schema(),
        )

    def _build_turn_detection_config(self, td: TurnDetection) -> TurnDetectionConfig:
        if isinstance(td, SemanticVAD):
            return SemanticVADConfig(eagerness=td.eagerness)
        return ServerVADConfig(
            threshold=td.threshold,
            prefix_padding_ms=td.prefix_padding_ms,
            silence_duration_ms=td.silence_duration_ms,
        )

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        if not self._is_connected():
            return

        await self._websocket.close()
        logger.info("Agent session stopped")
