import base64
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import ClassVar, override

import numpy as np
import pyaudio
import pygame

from rtvoice.sound.audio.strategy import AudioStrategy
from rtvoice.sound.models import AudioConfig, SoundFile
from rtvoice.state.base import VoiceAssistantEvent


class PyAudioStrategy(AudioStrategy):
    SUPPORTED_FORMATS: ClassVar[set[str]] = {".mp3"}
    DEFAULT_SOUNDS_DIR: ClassVar[Path] = Path(__file__).parent / "res"

    def __init__(
        self,
        config: AudioConfig | None = None,
        sounds_dir: str | Path | None = None,
    ):
        self._config = config or AudioConfig()
        self._pyaudio = pyaudio.PyAudio()
        self._stream: pyaudio.Stream | None = None
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._player_thread: threading.Thread | None = None
        self._current_audio_data = b""
        self._is_busy = False
        self._last_state_change = time.time()
        self._min_state_change_interval = 0.5
        self._stream_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._is_playing = False

        self.volume = 1.0

        self._sounds_dir = Path(sounds_dir) if sounds_dir else self.DEFAULT_SOUNDS_DIR
        self.logger.debug(
            "Initializing PyAudioStrategy with sounds directory: %s", self._sounds_dir
        )
        self._init_mixer()
        self._start_chunk_player()
        self.logger.debug("Chunk player auto-started during initialization")

    @override
    def clear_queue_and_stop_chunks(self) -> None:
        self.logger.debug("Clearing audio queue and stopping current chunk playback")

        with self._audio_queue.mutex:
            self._audio_queue.queue.clear()

        with self._stream_lock:
            if self._stream and self._stream.is_active():
                self._stream.stop_stream()
                time.sleep(0.05)
                self._stream.start_stream()

        with self._state_lock:
            if self._is_busy:
                self._is_busy = False
                self._current_audio_data = b""
                self._last_state_change = time.time()
                self._publish_event(VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED)

        self.logger.debug("Audio queue cleared, stream kept alive")

    @override
    def is_currently_playing_chunks(self) -> bool:
        current_time = time.time()

        with self._state_lock:
            if self._is_busy:
                return True

            if not self._audio_queue.empty():
                return True

            time_since_change = current_time - self._last_state_change
            if time_since_change < 0.3:
                return True

            with self._stream_lock:
                if self._stream:
                    return self._stream.is_active()

            return False

    @override
    def play_sound(self, sound_name: str) -> None:
        self._validate_audio_format(sound_name)
        sound_path = self._get_sound_path(sound_name)

        sound = pygame.mixer.Sound(sound_path)
        sound.set_volume(self.volume)
        sound.play()
        self.logger.debug("Playing sound: %s", sound_name)

    def _validate_audio_format(self, sound_name: str) -> None:
        if "." not in sound_name:
            return

        _, ext = os.path.splitext(sound_name.lower())
        if ext in self.SUPPORTED_FORMATS:
            return

        supported_list = ", ".join(self.SUPPORTED_FORMATS)
        raise ValueError(
            f"Audio format '{ext}' is not supported. "
            f"Supported formats: {supported_list}. "
            f"Please convert '{sound_name}' to MP3 format."
        )

    def _get_sound_path(self, sound_name: str) -> str:
        filename = sound_name if sound_name.endswith(".mp3") else f"{sound_name}.mp3"
        return str(self._sounds_dir / filename)

    @override
    def stop_sounds(self) -> None:
        self.logger.debug("Stopping all sound playback")

        if pygame.mixer.get_init():
            pygame.mixer.stop()
            self.logger.debug("Pygame mixer stopped")

        if self._player_thread and self._player_thread.is_alive():
            with self._stream_lock:
                if self._stream and self._stream.is_active():
                    self._stream.stop_stream()
                    self.logger.debug("Audio stream paused")

        self.clear_queue_and_stop_chunks()

    @override
    def get_volume_level(self) -> float:
        return self.volume

    @override
    def set_volume_level(self, volume: float) -> None:
        if not 0.0 <= volume <= 1.0:
            raise ValueError("Volume must be between 0.0 and 1.0")

        self.volume = volume
        self.logger.info("Volume set to: %.2f", self.volume)

    @override
    def play_sound_file(self, sound_file: SoundFile) -> None:
        self.play_sound(sound_file)

    @override
    def add_audio_chunk(self, base64_audio: str) -> None:
        try:
            audio_data = base64.b64decode(base64_audio)
            self._audio_queue.put(audio_data)
            self.logger.debug(
                "Added audio chunk to queue (size: %d bytes)", len(audio_data)
            )
        except Exception as e:
            self.logger.error("Error processing audio chunk: %s", e)

    def _start_chunk_player(self) -> None:
        self._is_playing = True
        with self._stream_lock:
            self._stream = self._pyaudio.open(
                format=self._config.format,
                channels=self._config.channels,
                rate=self._config.sample_rate,
                output=True,
                frames_per_buffer=self._config.chunk_size,
            )
        self._player_thread = threading.Thread(target=self._play_audio_loop)
        self._player_thread.daemon = True
        self._player_thread.start()
        self.logger.debug(
            "Audio chunk player started with sample rate: %d Hz",
            self._config.sample_rate,
        )

    def _play_audio_loop(self) -> None:
        while self._is_playing:
            try:
                chunk = self._get_next_audio_chunk()
                if not chunk:
                    continue

                self._process_audio_chunk(chunk)
                self._audio_queue.task_done()
                self._check_queue_state()

            except queue.Empty:
                continue
            except Exception as e:
                self._handle_playback_error(e)

    def _get_next_audio_chunk(self) -> bytes | None:
        try:
            return self._audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def _process_audio_chunk(self, chunk: bytes) -> None:
        if not chunk:
            return

        with self._state_lock:
            current_time = time.time()
            was_busy = self._is_busy
            self._is_busy = True

            if (
                not was_busy
                and (current_time - self._last_state_change)
                >= self._min_state_change_interval
            ):
                self._last_state_change = current_time
                self._publish_event(VoiceAssistantEvent.ASSISTANT_STARTED_RESPONDING)

        adjusted_chunk = self._adjust_volume(chunk)
        self._current_audio_data = adjusted_chunk

        try:
            with self._stream_lock:
                if self._stream and self._stream.is_active():
                    self._stream.write(adjusted_chunk)
                else:
                    self.logger.warning("Stream not active, skipping chunk")
                    self._recreate_audio_stream()
        except OSError as e:
            self.logger.error("Stream write error: %s", e)
            self._recreate_audio_stream()
        except Exception as e:
            self.logger.error("Unexpected error in stream write: %s", e)
            self._recreate_audio_stream()

    def _check_queue_state(self) -> None:
        with self._state_lock:
            if self._audio_queue.empty() and self._is_busy:
                current_time = time.time()

                if (
                    current_time - self._last_state_change
                ) >= self._min_state_change_interval:
                    self._is_busy = False
                    self._current_audio_data = b""
                    self._last_state_change = current_time
                    self._publish_event(
                        VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED
                    )

    def _handle_playback_error(self, error: Exception) -> None:
        error_traceback = traceback.format_exc()
        self.logger.error(
            "Error playing audio chunk: %s\nTraceback:\n%s", error, error_traceback
        )

        self._recreate_audio_stream()

        with self._state_lock:
            if self._is_busy:
                self._is_busy = False
                self._last_state_change = time.time()
                self._publish_event(VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED)

    def _adjust_volume(self, audio_chunk: bytes) -> bytes:
        if abs(self.volume - 1.0) < 1e-6:
            return audio_chunk

        try:
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            adjusted_array = (audio_array * self.volume).astype(np.int16)
            return adjusted_array.tobytes()
        except Exception as e:
            self.logger.error("Error adjusting volume: %s", e)
            return audio_chunk

    def _recreate_audio_stream(self) -> None:
        with self._stream_lock:
            if self._stream:
                self._stream.close()
                self._stream = None

            self._stream = self._pyaudio.open(
                format=self._config.format,
                channels=self._config.channels,
                rate=self._config.sample_rate,
                output=True,
                frames_per_buffer=self._config.chunk_size,
            )
            self.logger.debug("Audio stream recreated successfully")

    def _init_mixer(self) -> None:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            self.logger.debug("Pygame mixer initialized")
