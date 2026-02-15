import asyncio
from collections import deque
from collections.abc import Callable

from rtvoice.config.models import ModelSettings, VoiceSettings
from rtvoice.events.schemas import (
    AudioConfig,
    AudioFormat,
    AudioFormatConfig,
    AudioInputConfig,
    AudioOutputConfig,
    ConversationItemTruncatedEvent,
    ConversationResponseCreateEvent,
    RealtimeSessionConfig,
    SessionUpdateEvent,
)
from rtvoice.realtime.websocket.websocket_manager import WebSocketManager

from rtvoice.events import EventBus
from rtvoice.realtime.current_message_context import CurrentMessageContext
from rtvoice.shared.logging import LoggingMixin
from rtvoice.state.base import VoiceAssistantEvent
from rtvoice.tools.models import FunctionCallResult
from rtvoice.tools.registry.service import ToolRegistry


class RealtimeMessageManager(LoggingMixin):
    def __init__(
        self,
        ws_manager: WebSocketManager,
        tool_registry: ToolRegistry,
        model_settings: ModelSettings,
        voice_settings: VoiceSettings,
        event_bus: EventBus,
    ):
        self._ws_manager = ws_manager
        self._event_bus = event_bus
        self._model_settings = model_settings
        self._voice_settings = voice_settings
        self._tool_registry = tool_registry
        self._current_message_context = CurrentMessageContext(self._event_bus)

        self._response_active = False
        self._message_queue: deque = deque()
        self._queue_processing = False
        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        self._event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED,
            self._handle_speech_interruption,
        )
        self._event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_STARTED_RESPONSE,
            self._handle_response_started,
        )
        self._event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_COMPLETED_RESPONSE,
            self._handle_response_completed,
        )
        self._event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_CONFIG_UPDATE_REQUEST,
            self._handle_config_update_request,
        )

    async def initialize_session(self) -> None:
        if not self._ws_manager.is_connected():
            raise RuntimeError("No connection available for session initialization")

        session_config = self._build_config()
        self.logger.info("Sending session update...")
        await self._ws_manager.send_message(session_config)
        self.logger.info("Session initialized successfully")

    async def _send_or_queue(self, message_func: Callable, *args, **kwargs) -> None:
        if self._response_active:
            self.logger.debug("Response active - queueing message")
            self._message_queue.append((message_func, args, kwargs))
        else:
            self.logger.debug("No active response - sending message immediately")
            await message_func(*args, **kwargs)

    async def _process_queue(self) -> None:
        if self._queue_processing:
            return

        self._queue_processing = True
        try:
            while self._message_queue and not self._response_active:
                message_func, args, kwargs = self._message_queue.popleft()
                self.logger.debug("Processing queued message")
                await message_func(*args, **kwargs)
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error("Error processing message queue: %s", e, exc_info=True)
        finally:
            self._queue_processing = False

    async def send_tool_result(self, function_call_result: FunctionCallResult) -> None:
        await self._send_or_queue(self._send_tool_result_direct, function_call_result)

    async def send_loading_message(self, message: str) -> None:
        await self._send_or_queue(self._send_loading_message_direct, message)

    async def _send_tool_result_direct(
        self, function_call_result: FunctionCallResult
    ) -> None:
        self.logger.info("Sending tool result for '%s'", function_call_result.tool_name)

        await self._send_conversation_item(function_call_result)
        await self._trigger_response(function_call_result)

    async def _send_loading_message_direct(self, message: str) -> None:
        self.logger.info("Sending generator tool update")

        response_event = ConversationResponseCreateEvent.with_instructions(
            f"Say exactly: '{message}'. Do not add any information not in this message."
        )

        await self._ws_manager.send_message(response_event)

    async def _send_conversation_item(self, result: FunctionCallResult) -> None:
        conversation_item = result.to_conversation_item()
        await self._ws_manager.send_message(conversation_item)

    async def _trigger_response(self, result: FunctionCallResult) -> None:
        response_event = ConversationResponseCreateEvent.with_instructions(
            result.response_instruction
            or "Process the tool result and provide a helpful response."
        )
        await self._ws_manager.send_message(response_event)

    async def _send_session_update(self):
        session_config = self._build_config()
        self.logger.info("Sending session update with new config...")

        await self._ws_manager.send_message(session_config)

    def _build_config(self) -> SessionUpdateEvent:
        audio_config = AudioConfig(
            output=AudioOutputConfig(
                format=AudioFormatConfig(type=AudioFormat.PCM16),
                speed=self._voice_settings.speech_speed,
                voice=self._voice_settings.voice,
            ),
            input=AudioInputConfig(),
        )

        return SessionUpdateEvent(
            session=RealtimeSessionConfig(
                model=self._model_settings.model,
                instructions=self._model_settings.instructions,
                audio=audio_config,
                tools=self._tool_registry.get_openai_schema(),
            ),
        )

    async def _handle_config_update_request(self, new_response_speed: float, data=None):
        self.logger.info(
            "Received config update request - New response speed: %.2f",
            new_response_speed,
        )

        self._voice_settings.speech_speed = new_response_speed

        await self._send_session_update()

    async def _handle_speech_interruption(self, event=None, data=None) -> None:
        item_id = self._current_message_context.item_id
        duration_ms = self._current_message_context.current_duration_ms

        if not item_id or duration_ms is None:
            return

        self.logger.info("Truncating item %s at %d ms", item_id, duration_ms)

        truncate_event = ConversationItemTruncatedEvent(
            item_id=item_id,
            content_index=0,
            audio_end_ms=duration_ms,
        )

        await self._ws_manager.send_message(truncate_event)

    async def _handle_response_started(self, event=None, data=None) -> None:
        self._response_active = True

    async def _handle_response_completed(self, event=None, data=None) -> None:
        self._response_active = False
        await self._process_queue()
