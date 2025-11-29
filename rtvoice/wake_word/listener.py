import asyncio
import threading
from collections.abc import Mapping
from contextlib import suppress

import numpy as np
import pyaudio
from pvporcupine import (
    Porcupine,
    PorcupineInvalidArgumentError,
    PorcupineInvalidStateError,
    create,
)

from rtvoice.config import AgentEnv
from rtvoice.events import EventBus
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.state.base import VoiceAssistantEvent
from rtvoice.wake_word.models import PorcupineWakeWord


class WakeWordListener(LoggingMixin):
    def __init__(
        self,
        wakeword: PorcupineWakeWord,
        sensitivity: float,
        event_bus: EventBus,
        env: AgentEnv | None = None,
    ):
        self._detection_event = threading.Event()
        self.is_listening = False
        self.should_stop = False
        self.event_bus = event_bus

        self.wake_word = wakeword
        self.sensitivity = self._validate_sensitivity(sensitivity)

        self.logger.info(
            "Initializing Wake Word Listener with word=%s sensitivity=%.2f",
            self.wake_word,
            self.sensitivity,
        )

        self.env = env or AgentEnv()
        self.access_key = self.env.pico_access_key
        self.handle = self._create_handle(self.sensitivity)

        self.pa_input = pyaudio.PyAudio()
        self.stream = self._open_stream(self.handle.frame_length)

        self.logger.info("Wake Word Listener initialized")

    @staticmethod
    def _validate_sensitivity(sens: float) -> float:
        if not 0.0 <= sens <= 1.0:
            raise ValueError("sensitivity must be between 0.0 and 1.0")
        return float(sens)

    def __enter__(self):
        self.logger.debug("Entering WakeWordListener context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.debug("Exiting WakeWordListener context")
        self.cleanup()
        return False

    async def start(self) -> bool:
        self.logger.info("Starting async wake word listening…")
        self._detection_event.clear()
        self.should_stop = False
        self.is_listening = True

        if not self.stream.is_active():
            self.stream.start_stream()
            self.logger.info("Audio stream started")

        while not self.should_stop:
            if self._detection_event.is_set():
                self.logger.info("Wake word detected")
                self._detection_event.clear()
                self.is_listening = False
                return True
            await asyncio.sleep(0.1)

        self.logger.info("Wake word listening stopped")
        return False

    def stop(self) -> None:
        self.logger.info("Stopping wake word listener")
        self.should_stop = True
        self.is_listening = False

    def cleanup(self) -> None:
        self.logger.info("Cleaning up Wake Word Listener…")
        self.should_stop = True
        self.is_listening = False

        with suppress(Exception):
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()

        with suppress(Exception):
            if self.pa_input:
                self.pa_input.terminate()

        with suppress(Exception):
            if self.handle:
                self.handle.delete()

        self.logger.info("Wake Word Listener shut down")

    def _create_handle(self, sensitivity: float) -> Porcupine:
        handle = create(
            access_key=self.access_key,
            keywords=[self.wake_word],
            sensitivities=[sensitivity],
        )
        self.logger.info(
            "Porcupine handle created (word=%s, sens=%.2f)",
            self.wake_word,
            sensitivity,
        )
        return handle

    def _open_stream(self, frame_length: int):
        stream = self.pa_input.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=frame_length,
            stream_callback=self._audio_callback,
        )
        self.logger.info("Audio stream initialized (frame_length=%d)", frame_length)
        return stream

    def _audio_callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: Mapping[str, float],
        status: int,
    ) -> tuple[bytes | None, int]:
        if status:
            self.logger.warning("Audio callback status: %s", status)

        if self._should_process_audio(in_data):
            self._process_audio_frame(in_data)

        return (in_data, pyaudio.paContinue)

    def _should_process_audio(self, in_data: bytes | None) -> bool:
        return self.is_listening and not self.should_stop and in_data is not None

    def _process_audio_frame(self, in_data: bytes) -> None:
        try:
            pcm = np.frombuffer(in_data, dtype=np.int16)
            keyword_index = self.handle.process(pcm)

            if keyword_index >= 0:
                self._handle_wake_word_detected(keyword_index)
        except (
            ValueError,
            PorcupineInvalidStateError,
            PorcupineInvalidArgumentError,
        ) as e:
            self.logger.error("Audio processing error: %s", e)

    def _handle_wake_word_detected(self, keyword_index: int) -> None:
        self.logger.info("Wake word detected (index=%d)", keyword_index)
        self._detection_event.set()
        self.event_bus.publish_sync(VoiceAssistantEvent.WAKE_WORD_DETECTED)
