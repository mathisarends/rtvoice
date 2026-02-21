import asyncio
import time

from rtvoice.events import EventBus
from rtvoice.events.views import AudioPlaybackCompletedEvent, UserInactivityTimeoutEvent
from rtvoice.realtime.schemas import (
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ResponseCreatedEvent,
)
from rtvoice.shared.logging import LoggingMixin


# TODO: Hier zuverlässiger austimen (erst wenn das playback hier wirklich keine frames mehr liefert)
class UserInactivityTimeoutWatchdog(LoggingMixin):
    def __init__(self, event_bus: EventBus, timeout_seconds: float = 10.0):
        self.event_bus = event_bus
        self.timeout_seconds = timeout_seconds
        self._last_speech_time: float | None = None
        self._is_monitoring = False
        self._check_task: asyncio.Task | None = None
        self._assistant_is_speaking = False
        self._user_has_stopped_speaking = False

        self.event_bus.subscribe(
            InputAudioBufferSpeechStoppedEvent,
            self._handle_user_speech_ended,
        )
        self.event_bus.subscribe(
            InputAudioBufferSpeechStartedEvent,
            self._handle_user_started_speaking,
        )
        self.event_bus.subscribe(
            ResponseCreatedEvent,
            self._handle_assistant_started,
        )
        self.event_bus.subscribe(
            AudioPlaybackCompletedEvent,
            self._handle_assistant_done,
        )

    async def _handle_user_speech_ended(
        self, event: InputAudioBufferSpeechStoppedEvent
    ) -> None:
        self._user_has_stopped_speaking = True
        self.logger.debug(
            "User stopped speaking at %d ms",
            event.audio_end_ms,
        )

        self._try_start_monitoring()

    async def _handle_user_started_speaking(
        self, event: InputAudioBufferSpeechStartedEvent
    ) -> None:
        self._user_has_stopped_speaking = False
        self._is_monitoring = False
        self.logger.debug(
            "User started speaking at %d ms, stopping inactivity timeout monitoring",
            event.audio_start_ms,
        )

    async def _handle_assistant_started(self, event: ResponseCreatedEvent) -> None:
        self._assistant_is_speaking = True
        self._is_monitoring = False  # Stop monitoring while assistant speaks
        self.logger.debug("Assistant started speaking")

    async def _handle_assistant_done(self, event: AudioPlaybackCompletedEvent) -> None:
        self._assistant_is_speaking = False
        self.logger.debug("Assistant finished speaking")

        # Nur Timer starten, wenn User auch fertig ist
        self._try_start_monitoring()

    def _try_start_monitoring(self) -> None:
        if not self._user_has_stopped_speaking or self._assistant_is_speaking:
            return

        self._last_speech_time = time.monotonic()
        self._is_monitoring = True
        self.logger.debug(
            "Both user and assistant finished - starting inactivity timeout monitoring (%.1fs)",
            self.timeout_seconds,
        )

        if self._check_task is None or self._check_task.done():
            self._check_task = asyncio.create_task(self._monitor_timeout())

    async def _monitor_timeout(self) -> None:
        while self._is_monitoring:
            if not self._has_timed_out():
                await asyncio.sleep(0.5)
                continue

            self.logger.warning(
                "Inactivity timeout occurred after %.1f seconds",
                self.timeout_seconds,
            )
            await self.event_bus.dispatch(
                UserInactivityTimeoutEvent(timeout_seconds=self.timeout_seconds)
            )
            self._is_monitoring = False
            self._user_has_stopped_speaking = False
            break

    def _has_timed_out(self) -> bool:
        if self._last_speech_time is None:
            return False
        elapsed = time.monotonic() - self._last_speech_time
        return elapsed > self.timeout_seconds
