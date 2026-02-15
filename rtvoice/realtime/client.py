import asyncio
import base64

from rtvoice.config.models import ModelSettings, VoiceSettings
from rtvoice.events.schemas.audio import (
    InputAudioBufferAppendEvent,
)
from rtvoice.mic import MicrophoneCapture
from rtvoice.realtime.websocket.websocket_manager import WebSocketManager
from rtvoice.transcription import TranscriptionEventListener

from rtvoice.events import EventBus
from rtvoice.realtime.messaging.message_manager import RealtimeMessageManager
from rtvoice.shared.logging import LoggingMixin
from rtvoice.tools import (
    RemoteMcpToolEventListener,
    SpecialToolParameters,
    ToolExecutor,
    Tools,
)


class RealtimeClient(LoggingMixin):
    def __init__(
        self,
        model_settings: ModelSettings,
        voice_settings: VoiceSettings,
        audio_capture: MicrophoneCapture,
        special_tool_parameters: SpecialToolParameters,
        event_bus: EventBus,
        tools: Tools,
    ):
        self.model_settings = model_settings
        self.voice_settings = voice_settings
        self.audio_capture = audio_capture
        self.special_tool_parameters = special_tool_parameters
        self.event_bus = event_bus
        self.tools = tools
        self.tool_registry = self.tools.registry

        self._audio_player = special_tool_parameters.audio_player

        self.ws_manager = WebSocketManager.from_model(
            model=model_settings.model, event_bus=self.event_bus
        )
        self.transcription_service = TranscriptionEventListener(
            event_bus=self.event_bus
        )

        self.message_manager = RealtimeMessageManager(
            ws_manager=self.ws_manager,
            tool_registry=self.tool_registry,
            model_settings=model_settings,
            voice_settings=voice_settings,
            event_bus=self.event_bus,
        )

        self.tool_executor = ToolExecutor(
            self.tool_registry,
            self.message_manager,
            self.special_tool_parameters,
            self.event_bus,
        )

        self.mcp_tool_handler = RemoteMcpToolEventListener(
            event_bus=self.event_bus,
            message_manager=self.message_manager,
            ws_manager=self.ws_manager,
        )

        self._audio_streaming_paused = False
        self._audio_streaming_event = asyncio.Event()
        self._audio_streaming_event.set()

    async def setup_and_run(self) -> None:
        try:
            await self.ws_manager.create_connection()
            await self.message_manager.initialize_session()
            await self._send_audio_stream()
        except asyncio.CancelledError:
            self.logger.info("Audio streaming was cancelled")
        finally:
            await self.ws_manager.close()

    async def close_connection(self) -> None:
        self.logger.info("Closing WebSocket connection programmatically...")
        await self.ws_manager.close()
        self._audio_streaming_event.set()
        self.logger.info("WebSocket connection closed successfully")

    def pause_audio_streaming(self) -> None:
        self.logger.info("Pausing audio streaming...")
        self._audio_streaming_paused = True
        self._audio_streaming_event.clear()

    def resume_audio_streaming(self) -> None:
        self.logger.info("Resuming audio streaming...")
        self._audio_streaming_paused = False
        self._audio_streaming_event.set()

    def is_audio_streaming_paused(self) -> bool:
        return self._audio_streaming_paused

    async def _send_audio_stream(self) -> None:
        if not self.ws_manager.is_connected():
            raise RuntimeError("No connection available for audio transmission")

        self.logger.info("Starting audio transmission...")
        await self._process_audio_loop()
        self.logger.info("Audio transmission ended")

    async def _process_audio_loop(self) -> None:
        if not self.audio_capture.is_active:
            self.audio_capture.start_stream()

        try:
            async for chunk in self.audio_capture.stream_chunks():
                if not self._should_continue_streaming():
                    break

                await self._wait_for_streaming_resume()
                if self._audio_streaming_paused:
                    continue

                base64_audio_data = base64.b64encode(chunk).decode("utf-8")
                input_audio_buffer_append_event = (
                    InputAudioBufferAppendEvent.from_audio(base64_audio_data)
                )
                await self.ws_manager.send_message(input_audio_buffer_append_event)
        finally:
            self.audio_capture.stop_stream()

    def _should_continue_streaming(self) -> bool:
        return self.audio_capture.is_active and self.ws_manager.is_connected()

    async def _wait_for_streaming_resume(self) -> None:
        await self._audio_streaming_event.wait()
