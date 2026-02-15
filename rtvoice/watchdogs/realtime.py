from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentStartedEvent,
    AgentStoppedEvent,
    AssistantCompletedMCPToolCallResultEvent,
    AssistantFailedMCPToolCallEvent,
    MessageTruncationRequestedEvent,
)
from rtvoice.realtime.schemas import (
    ConversationItemTruncateEvent,
    ConversationResponseCreateEvent,
    InputAudioBufferAppendEvent,
    SessionUpdateEvent,
)
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.websocket import RealtimeWebSocket


class RealtimeWatchdog(LoggingMixin):
    def __init__(self, event_bus: EventBus, websocket: RealtimeWebSocket):
        self._event_bus = event_bus
        self._websocket = websocket

        self._event_bus.subscribe(AgentStartedEvent, self._on_agent_started)
        self._event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)
        self._event_bus.subscribe(
            InputAudioBufferAppendEvent, self._on_input_audio_buffer_append
        )
        self._event_bus.subscribe(
            AssistantCompletedMCPToolCallResultEvent,
            self._on_mcp_tool_call_completed,
        )
        self._event_bus.subscribe(
            AssistantFailedMCPToolCallEvent, self._on_mcp_tool_call_failed
        )
        self._event_bus.subscribe(
            MessageTruncationRequestedEvent, self._on_truncation_requested
        )

    def _is_connected(self) -> bool:
        return self._websocket.is_connected

    async def _on_input_audio_buffer_append(
        self, event: InputAudioBufferAppendEvent
    ) -> None:
        if not self._is_connected():
            self.logger.warning("WebSocket is not connected - cannot send audio data")
            return

        await self._websocket.send(event)
        self.logger.debug("Audio data sent to WebSocket")

    async def _on_agent_started(self, event: AgentStartedEvent) -> None:
        self.logger.info("Agent started - establishing WebSocket connection")

        if not self._is_connected():
            await self._websocket.connect()
            self.logger.info("WebSocket connection established successfully")

        self.logger.info("Initializing session with configuration...")
        session_update = SessionUpdateEvent(session=event.session_config)
        await self._websocket.send(session_update)
        self.logger.info("Session initialized successfully")

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        self.logger.info("Agent stopped - closing WebSocket connection")

        if not self._is_connected():
            self.logger.warning("WebSocket already disconnected")
            return

        await self._websocket.close()
        self.logger.info("WebSocket connection closed successfully")

    async def _on_mcp_tool_call_completed(
        self, _: AssistantCompletedMCPToolCallResultEvent
    ) -> None:
        self.logger.info("MCP tool call completed - sending response instruction")
        response_event = ConversationResponseCreateEvent.with_instructions(
            "MCP tool call has completed successfully. Please process the results "
            "and provide a response to the user."
        )
        await self._websocket.send(response_event)

    async def _on_mcp_tool_call_failed(
        self, _: AssistantFailedMCPToolCallEvent
    ) -> None:
        self.logger.info("MCP tool call failed - sending error instruction")
        response_event = ConversationResponseCreateEvent.with_instructions(
            "Something went wrong with the MCP tool call. Please inform the user "
            "about the issue."
        )
        await self._websocket.send(response_event)

    async def _on_truncation_requested(
        self, event: MessageTruncationRequestedEvent
    ) -> None:
        self.logger.info(
            "Truncation requested for item %s at %d ms",
            event.item_id,
            event.audio_end_ms,
        )

        truncate_event = ConversationItemTruncateEvent(
            item_id=event.item_id,
            content_index=0,
            audio_end_ms=event.audio_end_ms,
        )

        await self._websocket.send(truncate_event)
        self.logger.debug("Truncation event sent to WebSocket")
