from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from transitbus import EventBus

from rtvoice.agent.views import (
    InjectedAssistantMessage,
    InjectedConversation,
    InjectedUserMessage,
)
from rtvoice.audio import AudioSession
from rtvoice.events.views import (
    AgentSessionConnectedEvent,
    AgentStoppedEvent,
    InterruptAssistantCommand,
    UpdateSpeechSpeedCommand,
)
from rtvoice.handler import (
    AudioBridge,
    BargeInCoordinator,
    ConversationAudioRecorder,
    ConversationInactivityMonitor,
    SpeechActivityEventAdapter,
    ToolCallExecutor,
    TranscriptEventAdapter,
    TranscriptLogger,
)
from rtvoice.realtime.port import RealtimeProvider
from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
    SessionUpdateEvent,
    SpeedUpdateEvent,
)
from rtvoice.realtime.session_settings import (
    RealtimeSessionSettings,
    build_session_payload,
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
        settings: RealtimeSessionSettings,
        tools: Tools,
        audio_session: AudioSession,
        provider: RealtimeProvider,
        injected_conversation: InjectedConversation | None = None,
        inactivity_timeout_seconds: float | None = None,
        recording_path: Path | None = None,
        pricing_catalog: PricingCatalog | None = None,
    ):
        settings.model.warn_if_deprecated(stacklevel=3)
        self._event_bus = event_bus
        self._settings = settings
        self._tools = tools
        self._audio_session = audio_session
        self._injected_conversation = injected_conversation
        self._inactivity_timeout_seconds = inactivity_timeout_seconds
        self._recording_path = recording_path

        # settings are frozen; only the speed is retunable mid-session
        self._speech_speed = settings.speech_speed

        self._websocket = RealtimeWebSocket(model=settings.model, provider=provider)
        self._token_tracker = TokenTracker(
            event_bus=event_bus,
            realtime_model=settings.model.value,
            transcription_model=(
                settings.transcription_model.value
                if settings.transcription_model is not None
                else None
            ),
            pricing_catalog=pricing_catalog,
        )
        self._forward_task: asyncio.Task | None = None
        self._stopped = False
        self._setup_handlers()

        self._event_bus.on(AgentStoppedEvent, self._on_agent_stopped)
        self._event_bus.on(UpdateSpeechSpeedCommand, self._on_update_speech_speed)

    def _setup_handlers(self) -> None:
        self._transcript_logger = TranscriptLogger(event_bus=self._event_bus)
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

        if (
            self._settings.transcription_enabled
            or self._settings.assistant_text_enabled
        ):
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
    def settings(self) -> RealtimeSessionSettings:
        return self._settings

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
        logger.info("Applying session settings [%s]", self._settings.summary)
        settings = build_session_payload(self._settings, self._tools.get_schema())
        await self._websocket.send(SessionUpdateEvent(session=settings))

    @timed()
    async def _on_update_speech_speed(self, event: UpdateSpeechSpeedCommand) -> None:
        self._speech_speed = event.speed
        self._barge_in_coordinator.set_speech_speed(event.speed)

        if not self._websocket.is_connected:
            logger.warning("Cannot update speed - WebSocket not connected")
            return

        logger.info("Updating speech speed [speed=%s]", event.speed)
        await self._websocket.send(SpeedUpdateEvent.from_speed(event.speed))

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
