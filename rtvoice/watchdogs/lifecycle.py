import logging

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentStartedEvent,
    AgentStoppedEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioBufferAppendEvent,
    RealtimeSessionConfig,
    SessionUpdateEvent,
)
from rtvoice.realtime.websocket.service import RealtimeWebSocket

logger = logging.getLogger(__name__)


class LifecycleWatchdog:
    def __init__(self, event_bus: EventBus, websocket: RealtimeWebSocket):
        self._event_bus = event_bus
        self._websocket = websocket
        self._session_config: RealtimeSessionConfig | None = None

        self._event_bus.subscribe(AgentStartedEvent, self._on_agent_started)
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

    async def _on_agent_started(self, event: AgentStartedEvent) -> None:
        logger.info("Starting agent session")

        if not self._is_connected():
            await self._websocket.connect()

        self._session_config = event.session_config
        session_update = SessionUpdateEvent(session=self._session_config)
        await self._websocket.send(session_update)

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        if not self._is_connected():
            return

        await self._websocket.close()
        logger.info("Agent session stopped")
