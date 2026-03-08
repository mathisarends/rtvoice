import logging

from rtvoice.events import EventBus
from rtvoice.events.views import ConfigureSessionCommand, UpdateSpeechSpeedCommand
from rtvoice.realtime.schemas import (
    AudioConfig,
    AudioInputConfig,
    AudioOutputConfig,
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
from rtvoice.views import SemanticVAD, ServerVAD, TurnDetection

logger = logging.getLogger(__name__)


class SessionWatchdog:
    def __init__(self, event_bus: EventBus, websocket: RealtimeWebSocket):
        self._websocket = websocket
        event_bus.subscribe(ConfigureSessionCommand, self._on_configure_session)
        event_bus.subscribe(UpdateSpeechSpeedCommand, self._on_update_speech_speed)

    async def _on_configure_session(self, command: ConfigureSessionCommand) -> None:
        config = self._build_session_config(command)
        await self._websocket.send(SessionUpdateEvent(session=config))

    async def _on_update_speech_speed(self, command: UpdateSpeechSpeedCommand) -> None:
        if not self._websocket.is_connected:
            logger.warning("Cannot update speed — WebSocket not connected")
            return
        await self._websocket.send(SpeedUpdateEvent.from_speed(command.speed))

    def _build_session_config(
        self, command: ConfigureSessionCommand
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

    def _build_audio_input_config(
        self, command: ConfigureSessionCommand
    ) -> AudioInputConfig:
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
