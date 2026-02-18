import queue
import struct
import threading

import pyaudio

from rtvoice.audio.devices import AudioOutputDevice
from rtvoice.shared.logging import LoggingMixin


class SpeakerOutput(AudioOutputDevice, LoggingMixin):
    def __init__(self, device_index: int | None = None, sample_rate: int = 24000):
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._audio: pyaudio.PyAudio | None = None
        self._stream = None
        self._active = False
        self._volume = 1.0

        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._playback_thread: threading.Thread | None = None
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing or not self._queue.empty()

    async def start(self) -> None:
        if self._active:
            return

        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sample_rate,
            output=True,
            output_device_index=self._device_index,
        )
        self._active = True

        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True
        )
        self._playback_thread.start()

    async def stop(self) -> None:
        if not self._active:
            return

        self._active = False
        self._queue.put(None)  # sentinel

        if self._playback_thread:
            self._playback_thread.join(timeout=2.0)

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._audio:
            self._audio.terminate()

    def _playback_loop(self) -> None:
        while True:
            chunk = self._queue.get()
            if chunk is None:  # stop sentinel
                break
            self._playing = True
            if self._stream and self._active:
                self._stream.write(self._apply_volume(chunk))
            self._playing = False

    async def play_chunk(self, chunk: bytes) -> None:
        if not self._active:
            return
        self._queue.put(chunk)

    async def clear_buffer(self) -> None:
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        self.logger.debug("Cleared %d audio chunks from queue", cleared)

    async def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, volume))

    def _apply_volume(self, chunk: bytes) -> bytes:
        if self._volume == 1.0:
            return chunk
        sample_count = len(chunk) // 2
        samples = struct.unpack(f"<{sample_count}h", chunk)
        scaled = [int(s * self._volume) for s in samples]
        return struct.pack(f"<{sample_count}h", *scaled)
