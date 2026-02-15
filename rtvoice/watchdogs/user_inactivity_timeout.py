from __future__ import annotations

import asyncio
import time

from rtvoice.events import EventBus
from rtvoice.events.views import TimeoutOccurredEvent
from rtvoice.realtime.schemas import (
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
)
from rtvoice.shared.logging import LoggingMixin


class UserInactivityTimeoutWatchdog(LoggingMixin):
    def __init__(self, event_bus: EventBus, timeout_seconds: float = 10.0):
        self.event_bus = event_bus
        self.timeout_seconds = timeout_seconds
        self._last_speech_time: float | None = None
        self._is_monitoring = False
        self._check_task: asyncio.Task | None = None

        self.event_bus.subscribe(
            InputAudioBufferSpeechStoppedEvent,
            self._handle_user_speech_ended,
        )
        self.event_bus.subscribe(
            InputAudioBufferSpeechStartedEvent,
            self._handle_user_started_speaking,
        )

    async def _handle_user_speech_ended(
        self, event: InputAudioBufferSpeechStoppedEvent
    ) -> None:
        self._last_speech_time = time.monotonic()
        self._is_monitoring = True
        self.logger.debug(
            "User stopped speaking at %d ms, starting inactivity timeout monitoring (%.1fs)",
            event.audio_end_ms,
            self.timeout_seconds,
        )

        if self._check_task is None or self._check_task.done():
            self._check_task = asyncio.create_task(self._monitor_timeout())

    async def _handle_user_started_speaking(
        self, event: InputAudioBufferSpeechStartedEvent
    ) -> None:
        self._is_monitoring = False
        self.logger.debug(
            "User started speaking at %d ms, stopping inactivity timeout monitoring",
            event.audio_start_ms,
        )

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
                TimeoutOccurredEvent(timeout_seconds=self.timeout_seconds)
            )
            self._is_monitoring = False
            break

    def _has_timed_out(self) -> bool:
        if self._last_speech_time is None:
            return False
        elapsed = time.monotonic() - self._last_speech_time
        return elapsed > self.timeout_seconds

    def stop(self) -> None:
        self._is_monitoring = False
        if self._check_task and not self._check_task.done():
            self._check_task.cancel()
