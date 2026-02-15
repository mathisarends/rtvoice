from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentStartedEvent,
    AgentStoppedEvent,
    AssistantCompletedMCPToolCallResultEvent,
    AssistantFailedMCPToolCallEvent,
    ConversationItemCreateRequestedEvent,
    MessageTruncationRequestedEvent,
    SpeechSpeedUpdateRequestedEvent,
    ToolCallResultReadyEvent,
)
from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationItemTruncateEvent,
    ConversationResponseCreateEvent,
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
            AssistantCompletedMCPToolCallResultEvent,
            self._on_mcp_tool_call_completed,
        )
        self._event_bus.subscribe(
            AssistantFailedMCPToolCallEvent, self._on_mcp_tool_call_failed
        )
        self._event_bus.subscribe(
            MessageTruncationRequestedEvent, self._on_truncation_requested
        )
        self._event_bus.subscribe(
            SpeechSpeedUpdateRequestedEvent, self._on_speech_speed_update_requested
        )

        self._event_bus.subscribe(
            ConversationItemCreateRequestedEvent,
            self._on_conversation_item_create_requested,
        )
        self._event_bus.subscribe(
            ToolCallResultReadyEvent, self._on_tool_call_result_ready
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
        self._session_config = event.session_config
        session_update = SessionUpdateEvent(session=self._session_config)
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
        response_event = ConversationResponseCreateEvent.from_instructions(
            "MCP tool call has completed successfully. Process the results "
            "and provide a response to the user."
        )
        await self._websocket.send(response_event)

    async def _on_mcp_tool_call_failed(
        self, _: AssistantFailedMCPToolCallEvent
    ) -> None:
        self.logger.info("MCP tool call failed - sending error instruction")
        response_event = ConversationResponseCreateEvent.from_instructions(
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

    async def _on_speech_speed_update_requested(
        self, event: SpeechSpeedUpdateRequestedEvent
    ) -> None:
        if not self._session_config:
            self.logger.warning(
                "Cannot update speech speed - session config not initialized"
            )
            return

        clipped_speed = max(0.5, min(event.speech_speed, 1.5))
        rounded_speed = round(clipped_speed * 10) / 10

        if event.speech_speed != rounded_speed:
            self.logger.info(
                "Speech speed %.2f adjusted to %.1f",
                event.speech_speed,
                rounded_speed,
            )

        self._session_config.audio.output.speed = rounded_speed
        session_update = SessionUpdateEvent(session=self._session_config)
        await self._websocket.send(session_update)
        self.logger.info("Speech speed updated to %.1f", rounded_speed)

    async def _on_conversation_item_create_requested(
        self, event: ConversationItemCreateRequestedEvent
    ) -> None:
        self.logger.info("Creating conversation item")

        create_event = ConversationResponseCreateEvent.from_instructions(
            text=event.content
        )

        await self._websocket.send(create_event)

    async def _on_tool_call_result_ready(self, event: ToolCallResultReadyEvent) -> None:
        self.logger.info(
            "Tool call result ready: %s (call_id=%s)",
            event.tool_name,
            event.call_id,
        )

        create_event = ConversationItemCreateEvent.function_call_output(
            call_id=event.call_id,
            output=event.output,
        )
        await self._websocket.send(create_event)

        if event.response_instruction:
            response_event = ConversationResponseCreateEvent.from_instructions(
                event.response_instruction
            )
            await self._websocket.send(response_event)
        else:
            response_event = ConversationResponseCreateEvent.from_instructions(
                "The tool call has completed. Process the result and respond to the user."
            )
            await self._websocket.send(response_event)

        self.logger.debug("Conversation item created successfully")
