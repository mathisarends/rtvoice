import asyncio

from rtvoice.events.bus import EventBus
from rtvoice.mic.detector import SpeechDetector
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.state.base import VoiceAssistantEvent
from rtvoice.state.context import VoiceAssistantContext


class UserSpeechInactivityTimer(LoggingMixin):
    def __init__(
        self,
        timeout_seconds: float = 10.0,
        speech_detector: SpeechDetector | None = None,
    ):
        self._timeout_seconds = timeout_seconds
        self._speech_detector = speech_detector
        self._timeout_task: asyncio.Task | None = None
        self._event_bus: EventBus | None = None

    async def start(self, context: VoiceAssistantContext) -> None:
        self.logger.debug("Starting timer (%.1f seconds)", self._timeout_seconds)
        self._event_bus = context.event_bus

        if self._speech_detector:
            await self._speech_detector.start_monitoring(
                audio_capture=context.audio_capture,
                on_speech_detected=self._handle_speech_detected,
            )

        self._timeout_task = asyncio.create_task(
            self._timeout_loop(),
            name="user_speech_inactivity_timer",
        )

    async def stop(self) -> None:
        if self._timeout_task is None or self._timeout_task.done():
            return

        self.logger.debug("Stopping timer")

        if self._speech_detector:
            await self._speech_detector.stop_monitoring()

        self._timeout_task.cancel()
        try:
            await self._timeout_task
        except asyncio.CancelledError:
            pass
        finally:
            self._timeout_task = None
            self._event_bus = None

    def _handle_speech_detected(self) -> None:
        if self._timeout_task and not self._timeout_task.done():
            self.logger.debug("Speech detected, resetting timer")
            self._timeout_task.cancel()
            self._timeout_task = asyncio.create_task(
                self._timeout_loop(),
                name="user_speech_inactivity_timer",
            )

    async def _timeout_loop(self) -> None:
        try:
            await asyncio.sleep(self._timeout_seconds)
            self.logger.warning("Timeout occurred")

            if self._event_bus is None:
                raise RuntimeError(
                    "Event bus is not available for idle transition publish"
                )

            await self._event_bus.publish_async(VoiceAssistantEvent.IDLE_TRANSITION)

        except asyncio.CancelledError:
            self.logger.debug("Timeout cancelled")
            return
