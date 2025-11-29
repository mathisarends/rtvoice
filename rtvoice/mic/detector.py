import asyncio

import numpy as np

from rtvoice.events.bus import EventBus
from rtvoice.mic import MicrophoneCapture
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.state.base import VoiceAssistantEvent


class SpeechDetector(LoggingMixin):
    DEFAULT_THRESHOLD = 40.0
    DEFAULT_CHECK_INTERVAL = 0.1

    def __init__(
        self,
        audio_capture: MicrophoneCapture,
        event_bus: EventBus,
        threshold: float = DEFAULT_THRESHOLD,
        check_interval: float = DEFAULT_CHECK_INTERVAL,
    ):
        self.audio_capture = audio_capture
        self.event_bus = event_bus
        self.threshold = threshold
        self.check_interval = check_interval

        self._monitoring_task: asyncio.Task | None = None
        self._is_monitoring = False

    @property
    def is_monitoring(self) -> bool:
        return self._is_monitoring

    async def start_monitoring(self) -> None:
        if self._is_monitoring:
            self.logger.warning("Already monitoring")
            return

        self.logger.info("Starting speech detection (threshold: %.1f)", self.threshold)
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        if not self._is_monitoring:
            return

        self.logger.info("Stopping speech detection")
        self._is_monitoring = False

        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            finally:
                self._monitoring_task = None

    async def _monitor_loop(self) -> None:
        try:
            while self._is_monitoring:
                audio_data = self.audio_capture.read_chunk()

                if audio_data:
                    self._check_for_speech(audio_data)

                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            self.logger.debug("Monitoring cancelled")
        except Exception as e:
            self.logger.exception("Monitoring failed: %s", e)
            self._publish_error()

    def _check_for_speech(self, audio_data: bytes) -> None:
        audio_level = self._calculate_audio_level(audio_data)

        self.logger.debug(
            "Audio level: %.1f (threshold: %.1f)", audio_level, self.threshold
        )

        if audio_level > self.threshold:
            self.logger.info("Speech detected (level: %.1f)", audio_level)
            self._publish_speech_detected()

    def _calculate_audio_level(self, audio_data: bytes) -> float:
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        return float(np.sqrt(np.mean(audio_array**2)))

    def _publish_speech_detected(self) -> None:
        self.event_bus.publish_sync(VoiceAssistantEvent.USER_STARTED_SPEAKING)

    def _publish_error(self) -> None:
        self.event_bus.publish_sync(VoiceAssistantEvent.ERROR_OCCURRED)
