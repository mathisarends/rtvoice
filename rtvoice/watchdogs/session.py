import logging

from rtvoice.events import EventBus
from rtvoice.events.views import (
    ConfigureSessionCommand,
    UpdateSessionToolsCommand,
    UpdateSpeechSpeedCommand,
)
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
    ToolsUpdateEvent,
    TurnDetectionConfig,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.shared.decorators import timed
from rtvoice.views import SemanticVAD, ServerVAD, TurnDetection

logger = logging.getLogger(__name__)


class SessionWatchdog:
    def __init__(self, event_bus: EventBus, websocket: RealtimeWebSocket):
        self._websocket = websocket
        event_bus.subscribe(ConfigureSessionCommand, self._on_configure_session)
        event_bus.subscribe(UpdateSpeechSpeedCommand, self._on_update_speech_speed)
        event_bus.subscribe(UpdateSessionToolsCommand, self._on_update_session_tools)
        logger.debug("SessionWatchdog initialized, subscribed to 3 commands")

    @timed()
    async def _on_configure_session(self, command: ConfigureSessionCommand) -> None:
        logger.info(
            "Configuring session [model=%s, voice=%s, speed=%s, turn_detection=%s, transcription=%s]",
            command.model,
            command.voice,
            command.speech_speed,
            type(command.turn_detection).__name__,
            command.transcription_model,
        )
        config = self._build_session_config(command)
        await self._websocket.send(SessionUpdateEvent(session=config))
        logger.debug("SessionUpdateEvent sent")

    @timed()
    async def _on_update_speech_speed(self, command: UpdateSpeechSpeedCommand) -> None:
        if not self._websocket.is_connected:
            logger.warning("Cannot update speed — WebSocket not connected")
            return
        logger.info("Updating speech speed [speed=%s]", command.speed)
        await self._websocket.send(SpeedUpdateEvent.from_speed(command.speed))
        logger.debug("SpeedUpdateEvent sent")

    async def _on_update_session_tools(
        self, command: UpdateSessionToolsCommand
    ) -> None:
        tool_names = [t.name for t in command.tools]
        logger.info(
            "Updating session tools [count=%d, tools=%s]",
            len(command.tools),
            tool_names,
        )
        await self._websocket.send(ToolsUpdateEvent.from_tools(command.tools))
        logger.debug("ToolsUpdateEvent sent")

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
                output=self._build_audio_output_config(command),
            ),
        )

    def _build_audio_output_config(
        self, command: ConfigureSessionCommand
    ) -> AudioOutputConfig:
        return AudioOutputConfig(
            voice=command.voice.value,
            speed=command.speech_speed,
        )

    def _build_audio_input_config(
        self, command: ConfigureSessionCommand
    ) -> AudioInputConfig:
        return AudioInputConfig(
            turn_detection=self._build_turn_detection_config(command.turn_detection),
            noise_reduction=self._build_noise_reduction_config(command),
            transcription=self._build_transcription_config(command),
        )

    def _build_noise_reduction_config(
        self, command: ConfigureSessionCommand
    ) -> InputAudioNoiseReductionConfig:
        return InputAudioNoiseReductionConfig(
            type=NoiseReductionType(command.noise_reduction)
        )

    def _build_transcription_config(
        self, command: ConfigureSessionCommand
    ) -> InputAudioTranscriptionConfig | None:
        if command.transcription_model is None:
            return None
        return InputAudioTranscriptionConfig(model=command.transcription_model)

    def _build_turn_detection_config(
        self, turn_detection: TurnDetection
    ) -> TurnDetectionConfig:
        if isinstance(turn_detection, SemanticVAD):
            logger.debug("Using SemanticVAD [eagerness=%s]", turn_detection.eagerness)
            return SemanticVADConfig(eagerness=turn_detection.eagerness)

        if isinstance(turn_detection, ServerVAD):
            logger.debug(
                "Using ServerVAD [threshold=%s, prefix_padding_ms=%s, silence_duration_ms=%s]",
                turn_detection.threshold,
                turn_detection.prefix_padding_ms,
                turn_detection.silence_duration_ms,
            )
            return ServerVADConfig(
                threshold=turn_detection.threshold,
                prefix_padding_ms=turn_detection.prefix_padding_ms,
                silence_duration_ms=turn_detection.silence_duration_ms,
            )

        raise TypeError(f"Unknown TurnDetection type: {type(turn_detection)}")
