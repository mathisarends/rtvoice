from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Self

from rtvoice.audio import AudioSession
from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentSessionConnectedEvent,
    AgentStoppedEvent,
    UpdateSessionToolsCommand,
)
from rtvoice.handler import (
    AudioForwarder,
    AudioHandler,
    AudioRecorder,
    SpeechStateTracker,
    SubAgentCoordinator,
    ToolCallHandler,
    TranscriptionAccumulator,
)
from rtvoice.realtime.port import RealtimeProvider
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
from rtvoice.views import (
    AssistantVoice,
    NoiseReduction,
    OutputModality,
    RealtimeModel,
    SemanticVAD,
    ServerVAD,
    TranscriptionModel,
    TurnDetection,
)
from rtvoice.watchdogs.error import ErrorWatchdog
from rtvoice.watchdogs.interruption import InterruptionWatchdog
from rtvoice.watchdogs.user_inactivity_timeout import UserInactivityTimeoutWatchdog

if TYPE_CHECKING:
    from rtvoice.subagent import SubAgent
    from rtvoice.tools import Tools

logger = logging.getLogger(__name__)


class RealtimeSession:
    def __init__(
        self,
        *,
        event_bus: EventBus,
        model: RealtimeModel,
        instructions: str,
        voice: AssistantVoice,
        speech_speed: float,
        transcription_model: TranscriptionModel | None,
        output_modalities: list[OutputModality],
        noise_reduction: NoiseReduction,
        turn_detection: TurnDetection,
        tools: Tools,
        audio_session: AudioSession,
        subagents: list[SubAgent],
        inactivity_timeout_enabled: bool,
        inactivity_timeout_seconds: float | None,
        recording_path: Path | None,
        provider: RealtimeProvider,
    ):
        self._event_bus = event_bus
        self._model = model
        self._instructions = instructions
        self._voice = voice
        self._speech_speed = speech_speed
        self._transcription_model = transcription_model
        self._output_modalities = list(dict.fromkeys(output_modalities))
        self._noise_reduction = noise_reduction
        self._turn_detection = turn_detection
        self._tools = tools
        self._audio_session = audio_session
        self._subagents = subagents
        self._assistant_text_enabled = "text" in self._output_modalities
        self._transcription_enabled = self._transcription_model is not None
        self._inactivity_timeout_enabled = inactivity_timeout_enabled
        self._inactivity_timeout_seconds = inactivity_timeout_seconds
        self._recording_path = recording_path

        self._websocket = RealtimeWebSocket(model=model, provider=provider)
        self._forward_task: asyncio.Task | None = None
        self._stopped = False
        self._setup_handlers()

        self._event_bus.subscribe(
            UpdateSessionToolsCommand, self._on_update_session_tools
        )
        self._event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)

    def _setup_handlers(self) -> None:
        self._audio_handler = AudioHandler(
            event_bus=self._event_bus,
            audio_session=self._audio_session,
        )
        self._audio_forwarder = AudioForwarder(
            event_bus=self._event_bus,
            websocket=self._websocket,
        )
        self._interruption_watchdog = InterruptionWatchdog(
            event_bus=self._event_bus,
            websocket=self._websocket,
            audio_session=self._audio_session,
        )

        if self._transcription_enabled or self._assistant_text_enabled:
            self._transcription_accumulator = TranscriptionAccumulator(
                event_bus=self._event_bus
            )
            self._transcription_watchdog = self._transcription_accumulator

        self._tool_call_handler = ToolCallHandler(
            event_bus=self._event_bus,
            tools=self._tools,
            websocket=self._websocket,
            subagent_tool_names={s.name for s in self._subagents} or None,
        )

        if self._subagents:
            self._subagent_coordinator = SubAgentCoordinator(
                event_bus=self._event_bus,
                tools=self._tools,
                websocket=self._websocket,
            )
            for subagent in self._subagents:
                self._subagent_coordinator.register_subagent(subagent.name, subagent)

        self._error_watchdog = ErrorWatchdog(event_bus=self._event_bus)
        self._speech_state_tracker = SpeechStateTracker(event_bus=self._event_bus)

        if (
            self._inactivity_timeout_enabled
            and self._inactivity_timeout_seconds is not None
        ):
            self._user_inactivity_timeout_watchdog = UserInactivityTimeoutWatchdog(
                event_bus=self._event_bus,
                timeout_seconds=self._inactivity_timeout_seconds,
            )

        if self._recording_path:
            self._audio_recorder = AudioRecorder(
                event_bus=self._event_bus,
                output_path=self._recording_path,
            )

    @property
    def websocket(self) -> RealtimeWebSocket:
        return self._websocket

    async def __aenter__(self) -> Self:
        self._stopped = False
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._stop()

    @timed()
    async def start(self) -> None:
        logger.info("Starting realtime session")

        if not self._websocket.is_connected:
            await self._websocket.connect()

        if not self._forward_task or self._forward_task.done():
            self._forward_task = asyncio.create_task(self._forward_events())

        await self._send_session_update()
        await self._event_bus.dispatch(AgentSessionConnectedEvent())
        logger.info("Realtime session ready")

    async def _send_session_update(self) -> None:
        logger.info(
            "Configuring session [model=%s, voice=%s, speed=%s, turn_detection=%s, transcription=%s, output_modalities=%s]",
            self._model,
            self._voice,
            self._speech_speed,
            type(self._turn_detection).__name__,
            self._transcription_model,
            self._output_modalities,
        )
        config = self._build_session_config()
        await self._websocket.send(SessionUpdateEvent(session=config))

    @timed()
    async def update_speech_speed(self, speed: float) -> None:
        self._speech_speed = speed

        if not self._websocket.is_connected:
            logger.warning("Cannot update speed - WebSocket not connected")
            return

        logger.info("Updating speech speed [speed=%s]", speed)
        await self._websocket.send(SpeedUpdateEvent.from_speed(speed))

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

    async def _forward_events(self) -> None:
        async for event in self._websocket.events():
            await self._event_bus.dispatch(event)

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        await self._stop()

    async def _stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        if self._forward_task and not self._forward_task.done():
            self._forward_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._forward_task
        self._forward_task = None

        if not self._websocket.is_connected:
            return

        await self._websocket.close()
        logger.info("Realtime session stopped")

    def _build_session_config(
        self,
    ) -> RealtimeSessionConfig:
        if isinstance(self._turn_detection, SemanticVAD):
            turn_detection_config: TurnDetectionConfig = SemanticVADConfig(
                eagerness=self._turn_detection.eagerness
            )
        elif isinstance(self._turn_detection, ServerVAD):
            turn_detection_config = ServerVADConfig(
                threshold=self._turn_detection.threshold,
                prefix_padding_ms=self._turn_detection.prefix_padding_ms,
                silence_duration_ms=self._turn_detection.silence_duration_ms,
            )
        else:
            raise TypeError(f"Unknown TurnDetection type: {type(self._turn_detection)}")

        transcription_config = (
            None
            if self._transcription_model is None
            else InputAudioTranscriptionConfig(model=self._transcription_model)
        )

        return RealtimeSessionConfig(
            model=self._model,
            instructions=self._instructions,
            output_modalities=self._output_modalities,
            tool_choice=ToolChoiceMode.AUTO,
            tools=self._tools.get_tool_schema(),
            audio=AudioConfig(
                input=AudioInputConfig(
                    turn_detection=turn_detection_config,
                    noise_reduction=InputAudioNoiseReductionConfig(
                        type=NoiseReductionType(self._noise_reduction)
                    ),
                    transcription=transcription_config,
                ),
                output=AudioOutputConfig(
                    voice=self._voice.value, speed=self._speech_speed
                ),
            ),
        )
