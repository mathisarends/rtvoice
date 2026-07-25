from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from transitbus import EventBus

from rtvoice.agent.views import (
    AssistantVoice,
    InjectedAssistantMessage,
    InjectedConversation,
    InjectedUserMessage,
    NoiseReduction,
    OutputModality,
    RealtimeModel,
    ReasoningEffort,
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
    InputAudioTranscriptionSettings,
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
        injected_conversation: InjectedConversation | None,
        inactivity_timeout_seconds: float | None,
        recording_path: Path | None,
        provider: RealtimeProvider,
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
        self._injected_conversation = injected_conversation
        self._assistant_text_enabled = "text" in self._output_modalities
        self._transcription_enabled = self._transcription_model is not None
        self._inactivity_timeout_seconds = inactivity_timeout_seconds
        self._recording_path = recording_path

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
            speech_speed=self._speech_speed,
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
    def recorpding_path(self) -> Path | None:
        return self._recording_path

    @property
    def usage_report(self) -> UsageReport:
        return self._token_tracker.report()

    @timed()
    async def start(self) -> None:
        logger.info("Starting realtime session")

        if not self._websocket.is_connected:
            await self._websocket.connect()

        if not self._forward_task or self._forward_task.done():
            self._forward_task = asyncio.create_task(self._forward_events())

        await self._send_session_update()
        await self._send_injected_conversation()
        await self._event_bus.dispatch(AgentSessionConnectedEvent())
        logger.info("Realtime session ready")

    async def _send_injected_conversation(self) -> None:
        if not self._injected_conversation:
            return

        logger.info(
            "Injecting conversation [messages=%d]",
            len(self._injected_conversation.messages),
        )
        for message in self._injected_conversation.messages:
            await self._websocket.send(self._injected_message_event(message))

    def _injected_message_event(
        self, message: InjectedUserMessage | InjectedAssistantMessage
    ) -> ConversationItemCreateEvent:
        if isinstance(message, InjectedUserMessage):
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
        self._barge_in_coordinator.set_speech_speed(speed)

        if not self._websocket.is_connected:
            logger.warning("Cannot update speed - WebSocket not connected")
            return

        logger.info("Updating speech speed [speed=%s]", speed)
        await self._websocket.send(SpeedUpdateEvent.from_speed(speed))

    async def interrupt(self) -> None:
        await self._event_bus.dispatch(InterruptAssistantCommand())

    async def send_message(self, text: str, *, base64_image: str | None = None) -> bool:
        if not self._websocket.is_connected:
            logger.warning("Cannot send message - WebSocket not connected")
            return False

        logger.info(
            "Sending user message [text=%r, image=%s]", text, bool(base64_image)
        )
        item = (
            ConversationItemCreateEvent.user_message_with_image(text, base64_image)
            if base64_image
            else ConversationItemCreateEvent.user_message(text)
        )
        await self._websocket.send(item)
        await self._websocket.send(ConversationResponseCreateEvent())
        return True

    async def send_assistant_message(self, text: str) -> bool:
        if not self._websocket.is_connected:
            logger.warning("Cannot send assistant message - WebSocket not connected")
            return False

        logger.info("Injecting assistant message [text=%r]", text)
        await self._websocket.send(ConversationItemCreateEvent.assistant_message(text))
        return True

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
        match self._turn_detection:
            case SemanticVAD(eagerness=eagerness):
                turn_detection_settings: TurnDetectionSettings = SemanticVADSettings(
                    eagerness=eagerness
                )
            case ServerVAD(
                threshold=threshold,
                prefix_padding_ms=prefix_padding_ms,
                silence_duration_ms=silence_duration_ms,
            ):
                turn_detection_settings = ServerVADSettings(
                    threshold=threshold,
                    prefix_padding_ms=prefix_padding_ms,
                    silence_duration_ms=silence_duration_ms,
                )

        transcription_settings = (
            None
            if self._transcription_model is None
            else InputAudioTranscriptionSettings(model=self._transcription_model)
        )

        return RealtimeSessionSettings(
            model=self._model,
            reasoning=(
                None
                if self._reasoning_effort is None
                else {"effort": self._reasoning_effort}
            ),
            instructions=self._instructions,
            output_modalities=self._output_modalities,
            tool_choice=ToolChoiceMode.AUTO,
            tools=self._tools.get_schema(),
            audio=AudioSettings(
                input=AudioInputSettings.with_noise_reduction(
                    turn_detection=turn_detection_settings,
                    noise_reduction=self._noise_reduction,
                    transcription=transcription_settings,
                ),
                output=AudioOutputSettings(
                    voice=self._voice.value, speed=self._speech_speed
                ),
            ),
        )
