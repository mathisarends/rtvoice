import asyncio
import struct

import pyaudio

from rtvoice.audio.devices import AudioOutputDevice
from rtvoice.shared.logging import LoggingMixin


class SpeakerOutput(AudioOutputDevice, LoggingMixin):
    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int = 24000,
    ):
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._audio: pyaudio.PyAudio | None = None
        self._stream = None
        self._active = False
        self._volume = 1.0

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

    async def stop(self) -> None:
        if not self._active:
            return

        self._active = False

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._audio:
            self._audio.terminate()
            self._audio = None

    async def play_chunk(self, chunk: bytes) -> None:
        if not self._active or not self._stream:
            return

        scaled_chunk = self._apply_volume(chunk)
        await asyncio.get_event_loop().run_in_executor(
            None, self._stream.write, scaled_chunk
        )

    async def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, volume))
        self.logger.debug("Volume set to %.2f", self._volume)

    def _apply_volume(self, chunk: bytes) -> bytes:
        if self._volume == 1.0:
            return chunk

        sample_count = len(chunk) // 2
        samples = struct.unpack(f"<{sample_count}h", chunk)
        scaled_samples = [int(sample * self._volume) for sample in samples]
        return struct.pack(f"<{sample_count}h", *scaled_samples)

    async def clear_buffer(self) -> None:
        """Clear PyAudio's internal buffer by stopping and restarting the stream."""
        if not self._active or not self._stream:
            return

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._stream.stop_stream
            )

            await asyncio.get_event_loop().run_in_executor(
                None, self._stream.start_stream
            )
        except Exception as e:
            self.logger.warning(f"Error clearing audio buffer: {e}")
