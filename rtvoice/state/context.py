from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rtvoice.events import EventBus
from rtvoice.mic import MicrophoneCapture, SpeechDetector
from rtvoice.realtime.reatlime_client import RealtimeClient
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.sound import AudioPlayer
from rtvoice.state.base import VoiceAssistantEvent

if TYPE_CHECKING:
    from rtvoice.state.base import AssistantState
    from rtvoice.wake_word import WakeWordListener


class VoiceAssistantContext(LoggingMixin):
    def __init__(
        self,
        wake_word_listener: WakeWordListener,
        audio_capture: MicrophoneCapture,
        speech_detector: SpeechDetector,
        audio_player: AudioPlayer,
        event_bus: EventBus,
        realtime_client: RealtimeClient,
    ):
        from rtvoice.state.idle import IdleState

        self._state: AssistantState = IdleState()
        self._wake_word_listener = wake_word_listener
        self._audio_capture = audio_capture
        self._speech_detector = speech_detector
        self._audio_player = audio_player
        self._event_bus = event_bus
        self._realtime_client = realtime_client

        self._is_first_state_machine_loop_after_startup = True
        self._realtime_task: asyncio.Task | None = None

        self._setup_event_subscriptions()

    @property
    def state(self) -> AssistantState:
        return self._state

    @state.setter
    def state(self, new_state: AssistantState) -> None:
        self._state = new_state

    @property
    def wake_word_listener(self) -> WakeWordListener:
        return self._wake_word_listener

    @property
    def audio_capture(self) -> MicrophoneCapture:
        return self._audio_capture

    @property
    def speech_detector(self) -> SpeechDetector:
        return self._speech_detector

    @property
    def audio_player(self) -> AudioPlayer:
        return self._audio_player

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def realtime_client(self) -> RealtimeClient:
        return self._realtime_client

    async def run(self) -> None:
        self._audio_player.play_startup_sound()
        await self._state.on_enter(self)

    def _setup_event_subscriptions(self) -> None:
        for event_type in VoiceAssistantEvent:
            self._event_bus.subscribe(event_type, self.handle_event)

    async def handle_event(self, event: VoiceAssistantEvent, data: Any = None) -> None:
        if event == VoiceAssistantEvent.IDLE_TRANSITION:
            await self._state.transition_to_idle(self)
        else:
            await self._state.handle(event, self)

    async def start_realtime_session(self) -> None:
        if self._is_realtime_session_active():
            self.logger.warning("Realtime session already active, skipping start")
            return

        self.logger.info("Starting realtime session...")
        self._realtime_task = asyncio.create_task(self._realtime_client.setup_and_run())
        self.logger.info("Realtime session started successfully")
        return True

    async def close_realtime_session(self, timeout: float = 1.0) -> bool:
        if not self._is_realtime_session_active():
            return True

        try:
            await self._realtime_client.close_connection()

            if self._realtime_task:
                await asyncio.wait_for(self._realtime_task, timeout=timeout)

            self._realtime_task = None
            return True

        except TimeoutError:
            self.logger.error("Task didn't complete - this should not happen!")
            self._realtime_task.cancel()
            return False

    def ensure_realtime_audio_channel_paused(self) -> None:
        if not self._is_realtime_session_active():
            raise RuntimeError("Cannot pause audio - realtime session not active")

        if not self._is_realtime_audio_paused():
            self._realtime_client.pause_audio_streaming()
            self.logger.info("Realtime audio streaming paused")

    async def ensure_realtime_audio_channel_connected(self) -> None:
        if not self._is_realtime_session_active():
            self.logger.info("Realtime session not active, starting new session...")
            success = await self.start_realtime_session()
            if not success:
                raise RuntimeError("Failed to start realtime session")

        if not self._audio_capture.is_active:
            self._audio_capture.start_stream()
            self.logger.info("Microphone stream reactivated")

        if self._is_realtime_audio_paused():
            self._realtime_client.resume_audio_streaming()
            self.logger.info("Realtime audio streaming resumed")

    def _is_realtime_audio_paused(self) -> bool:
        if not self._is_realtime_session_active():
            return False
        return self._realtime_client.is_audio_streaming_paused()

    def _is_realtime_session_active(self) -> bool:
        return self._realtime_task is not None and not self._realtime_task.done()
