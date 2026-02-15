from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentStartedEvent,
    AgentStoppedEvent,
    SpeechSpeedUpdateRequestedEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioBufferAppendEvent,
    RealtimeSessionConfig,
    SessionUpdateEvent,
)
from rtvoice.realtime.websocket.service import RealtimeWebSocket
from rtvoice.shared.logging import LoggingMixin


class RealtimeWatchdog(LoggingMixin):
    def __init__(self, event_bus: EventBus, websocket: RealtimeWebSocket):
        self._event_bus = event_bus
        self._websocket = websocket
        self._session_config: RealtimeSessionConfig | None = None

        self._event_bus.subscribe(AgentStartedEvent, self._on_agent_started)
        self._event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)
        self._event_bus.subscribe(
            InputAudioBufferAppendEvent, self._on_input_audio_buffer_append
        )
        self._event_bus.subscribe(
            SpeechSpeedUpdateRequestedEvent, self._on_speech_speed_update_requested
        )

    def _is_connected(self) -> bool:
        return self._websocket.is_connected

    async def _on_input_audio_buffer_append(
        self, event: InputAudioBufferAppendEvent
    ) -> None:
        if not self._is_connected():
            self.logger.warning("Cannot send audio - WebSocket not connected")
            return

        await self._websocket.send(event)

    async def _on_agent_started(self, event: AgentStartedEvent) -> None:
        self.logger.info("Starting agent session")

        if not self._is_connected():
            await self._websocket.connect()

        self._session_config = event.session_config
        session_update = SessionUpdateEvent(session=self._session_config)
        await self._websocket.send(session_update)

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        if not self._is_connected():
            return

        await self._websocket.close()
        self.logger.info("Agent session stopped")

    async def _on_speech_speed_update_requested(
        self, event: SpeechSpeedUpdateRequestedEvent
    ) -> None:
        if not self._session_config:
            self.logger.warning("Cannot update speech speed - no active session")
            return

        clipped_speed = max(0.5, min(event.speech_speed, 1.5))
        rounded_speed = round(clipped_speed * 10) / 10

        if event.speech_speed != rounded_speed:
            self.logger.debug(
                "Speech speed %.2f adjusted to %.1f",
                event.speech_speed,
                rounded_speed,
            )

        self._session_config.audio.output.speed = rounded_speed
        session_update = SessionUpdateEvent(session=self._session_config)
        await self._websocket.send(session_update)
