import asyncio
from collections.abc import Callable

import numpy as np

from rtvoice.mic import MicrophoneCapture
from rtvoice.shared.logging_mixin import LoggingMixin


class SpeechDetector(LoggingMixin):
    DEFAULT_THRESHOLD = 60.0
    DEFAULT_CHECK_INTERVAL = 0.1

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        check_interval: float = DEFAULT_CHECK_INTERVAL,
    ):
        self.threshold = threshold
        self.check_interval = check_interval

        self._monitoring_task: asyncio.Task | None = None
        self._is_monitoring = False
        self._audio_capture: MicrophoneCapture | None = None
        self._on_speech_detected: Callable[[], None] | None = None

    @property
    def is_monitoring(self) -> bool:
        return self._is_monitoring

    async def start_monitoring(
        self,
        audio_capture: MicrophoneCapture,
        on_speech_detected: Callable[[], None],
    ) -> None:
        if self._is_monitoring:
            self.logger.warning("Already monitoring")
            return

        self.logger.info("Starting speech detection (threshold: %.1f)", self.threshold)
        self._audio_capture = audio_capture
        self._on_speech_detected = on_speech_detected
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
                self._audio_capture = None
                self._on_speech_detected = None

    async def _monitor_loop(self) -> None:
        try:
            while self._is_monitoring and self._audio_capture:
                audio_data = self._audio_capture.read_chunk()

                if audio_data:
                    self._check_for_speech(audio_data)

                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            self.logger.debug("Monitoring cancelled")
        except Exception as e:
            self.logger.exception("Monitoring failed: %s", e)

    def _check_for_speech(self, audio_data: bytes) -> None:
        audio_level = self._calculate_audio_level(audio_data)

        self.logger.debug(
            "Audio level: %.1f (threshold: %.1f)", audio_level, self.threshold
        )

        if audio_level > self.threshold and self._on_speech_detected:
            self.logger.info("Speech detected (level: %.1f)", audio_level)
            self._on_speech_detected()

    def _calculate_audio_level(self, audio_data: bytes) -> float:
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        return float(np.sqrt(np.mean(audio_array**2)))
