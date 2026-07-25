from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from transitbus import EventBus

from rtvoice.agent.views import (
    AssistantVoice,
    ConversationSeed,
    NoiseReduction,
    OutputModality,
    RealtimeModel,
    ReasoningEffort,
    SeedMessage,
    SemanticVAD,
    ServerVAD,
    TranscriptionModel,
    TurnDetection,
)
from rtvoice.audio import AudioSession
from rtvoice.events.views import (
    AgentSessionConnectedEvent,
    AgentStoppedEvent,
    InterruptAssistantCommand,
)
from rtvoice.handler import (
    AudioBridge,
    BargeInCoordinator,
    ConversationAudioRecorder,
    ConversationInactivityMonitor,
    SpeechActivityEventAdapter,
    ToolCallExecutor,
    TranscriptEventAdapter,
)
from rtvoice.realtime.port import RealtimeProvider
from rtvoice.realtime.schemas import (
    AudioInputSettings,
    AudioOutputSettings,
    AudioSettings,
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
    InputAudioNoiseReductionSettings,
    InputAudioTranscriptionSettings,
    NoiseReductionType,
    RealtimeSessionSettings,
    SemanticVADSettings,
    ServerVADSettings,
    SessionUpdateEvent,
    SpeedUpdateEvent,
    ToolChoiceMode,
    TurnDetectionSettings,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.shared.decorators import timed
from rtvoice.tokens.models import UsageReport
from rtvoice.tokens.pricing import PricingCatalog
from rtvoice.tokens.tracker import TokenTracker
from rtvoice.watchdogs import ErrorWatchdog

if TYPE_CHECKING:
    from rtvoice.tools import Tools

logger = logging.getLogger(__name__)

_PREAMBLE_GUIDANCE = (
    "When you are about to call a tool that may take a moment, first say a short, "
    "natural acknowledgment of what you are doing. If the user asks for a result "
    "that is not ready yet, tell them you are still working on it."
)


class RealtimeSession:
    def __init__(
        self,
        *,
        event_bus: EventBus,
        model: RealtimeModel,
        reasoning_effort: ReasoningEffort | None,
        instructions: str,
        voice: AssistantVoice,
        speech_speed: float,
        transcription_model: TranscriptionModel | None,
        output_modalities: list[OutputModality],
        noise_reduction: NoiseReduction,
        turn_detection: TurnDetection,
        tools: Tools,
        audio_session: AudioSession,
        conversation_seed: ConversationSeed | None,
        inactivity_timeout_seconds: float | None,
        recording_path: Path | None,
        provider: RealtimeProvider,
        enable_preambles: bool = True,
        pricing_catalog: PricingCatalog | None = None,
    ):
        model.warn_if_deprecated(stacklevel=3)
        self._event_bus = event_bus
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._instructions = instructions
        self._voice = voice
        self._speech_speed = speech_speed
        self._transcription_model = transcription_model
        self._output_modalities = list(dict.fromkeys(output_modalities))
        self._noise_reduction = noise_reduction
        self._turn_detection = turn_detection
        self._tools = tools
        self._audio_session = audio_session
        self._conversation_seed = conversation_seed
        self._assistant_text_enabled = "text" in self._output_modalities
        self._transcription_enabled = self._transcription_model is not None
        self._inactivity_timeout_seconds = inactivity_timeout_seconds
        self._recording_path = recording_path
        self._enable_preambles = enable_preambles

        self._websocket = RealtimeWebSocket(model=model, provider=provider)
        self._token_tracker = TokenTracker(
            event_bus=event_bus,
            realtime_model=model.value,
            transcription_model=(
                transcription_model.value if transcription_model is not None else None
            ),
            pricing_catalog=pricing_catalog,
        )
        self._forward_task: asyncio.Task | None = None
        self._stopped = False
        self._setup_handlers()

        self._event_bus.on(AgentStoppedEvent, self._on_agent_stopped)

    def _setup_handlers(self) -> None:
        self._audio_bridge = AudioBridge(
            event_bus=self._event_bus,
            audio_session=self._audio_session,
            websocket=self._websocket,
        )
        self._barge_in_coordinator = BargeInCoordinator(
            event_bus=self._event_bus,
            websocket=self._websocket,
            audio_session=self._audio_session,
        )

        if self._transcription_enabled or self._assistant_text_enabled:
            self._transcript_event_adapter = TranscriptEventAdapter(
                event_bus=self._event_bus
            )

        self._tool_call_executor = ToolCallExecutor(
            event_bus=self._event_bus,
            tools=self._tools,
            websocket=self._websocket,
        )

        self._error_watchdog = ErrorWatchdog(event_bus=self._event_bus)
        self._speech_activity_event_adapter = SpeechActivityEventAdapter(
            event_bus=self._event_bus
        )

        if self._inactivity_timeout_seconds is not None:
            self._conversation_inactivity_monitor = ConversationInactivityMonitor(
                event_bus=self._event_bus,
                timeout_seconds=self._inactivity_timeout_seconds,
            )

        if self._recording_path:
            self._conversation_audio_recorder = ConversationAudioRecorder(
                event_bus=self._event_bus,
                output_path=self._recording_path,
            )

    @property
    def recording_path(self) -> Path | None:
        return self._recording_path

    @property
    def usage_report(self) -> UsageReport:
        return self._token_tracker.report()

    @property
    def token_tracker(self) -> TokenTracker:
        return self._token_tracker

    @timed()
    async def start(self) -> None:
        logger.info("Starting realtime session")

        if not self._websocket.is_connected:
            await self._websocket.connect()

        if not self._forward_task or self._forward_task.done():
            self._forward_task = asyncio.create_task(self._forward_events())

        await self._send_session_update()
        await self._inject_conversation_seed()
        await self._event_bus.dispatch(AgentSessionConnectedEvent())
        logger.info("Realtime session ready")

    async def _inject_conversation_seed(self) -> None:
        if not self._conversation_seed:
            return

        logger.info(
            "Injecting conversation seed [messages=%d]",
            len(self._conversation_seed.messages),
        )
        for message in self._conversation_seed.messages:
            await self._websocket.send(self._seed_message_event(message))

    def _seed_message_event(self, message: SeedMessage) -> ConversationItemCreateEvent:
        if message.role == "user":
            return ConversationItemCreateEvent.user_message(message.text)
        return ConversationItemCreateEvent.assistant_message(message.text)

    async def _send_session_update(self) -> None:
        logger.info(
            "Applying session settings [model=%s, reasoning_effort=%s, voice=%s, speed=%s, turn_detection=%s, transcription=%s, output_modalities=%s]",
            self._model,
            self._reasoning_effort,
            self._voice,
            self._speech_speed,
            type(self._turn_detection).__name__,
            self._transcription_model,
            self._output_modalities,
        )
        settings = self._build_session_settings()
        await self._websocket.send(SessionUpdateEvent(session=settings))

    @timed()
    async def update_speech_speed(self, speed: float) -> None:
        self._speech_speed = speed

        if not self._websocket.is_connected:
            logger.warning("Cannot update speed - WebSocket not connected")
            return

        logger.info("Updating speech speed [speed=%s]", speed)
        await self._websocket.send(SpeedUpdateEvent.from_speed(speed))

    async def interrupt(self) -> None:
        await self._event_bus.dispatch(InterruptAssistantCommand())

    async def send_image(self, image_data_url: str, text: str = "") -> None:
        if not self._websocket.is_connected:
            logger.warning("Cannot send image - WebSocket not connected")
            return

        logger.info("Sending image input [text=%r]", text)
        await self._websocket.send(
            ConversationItemCreateEvent.user_message_with_image(text, image_data_url)
        )
        await self._websocket.send(ConversationResponseCreateEvent())

    async def _forward_events(self) -> None:
        async for event in self._websocket.events():
            await self._event_bus.dispatch(event)

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        await self.stop()

    async def stop(self) -> None:
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

    def _build_session_settings(
        self,
    ) -> RealtimeSessionSettings:
        if isinstance(self._turn_detection, SemanticVAD):
            turn_detection_settings: TurnDetectionSettings = SemanticVADSettings(
                eagerness=self._turn_detection.eagerness
            )
        elif isinstance(self._turn_detection, ServerVAD):
            turn_detection_settings = ServerVADSettings(
                threshold=self._turn_detection.threshold,
                prefix_padding_ms=self._turn_detection.prefix_padding_ms,
                silence_duration_ms=self._turn_detection.silence_duration_ms,
            )
        else:
            raise TypeError(f"Unknown TurnDetection type: {type(self._turn_detection)}")

        transcription_settings = (
            None
            if self._transcription_model is None
            else InputAudioTranscriptionSettings(model=self._transcription_model)
        )

        instructions = self._instructions
        if self._enable_preambles and self._tools.get_tool_schema():
            instructions = "\n\n".join(
                part for part in (instructions, _PREAMBLE_GUIDANCE) if part
            )

        return RealtimeSessionSettings(
            model=self._model,
            reasoning=(
                None
                if self._reasoning_effort is None
                else {"effort": self._reasoning_effort}
            ),
            instructions=instructions,
            output_modalities=self._output_modalities,
            tool_choice=ToolChoiceMode.AUTO,
            tools=self._tools.get_tool_schema(),
            audio=AudioSettings(
                input=AudioInputSettings(
                    turn_detection=turn_detection_settings,
                    noise_reduction=InputAudioNoiseReductionSettings(
                        type=NoiseReductionType(self._noise_reduction)
                    ),
                    transcription=transcription_settings,
                ),
                output=AudioOutputSettings(
                    voice=self._voice.value, speed=self._speech_speed
                ),
            ),
        )
