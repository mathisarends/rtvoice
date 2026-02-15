import asyncio
from collections.abc import AsyncIterator

import pyaudio

from rtvoice.audio.devices import AudioInputDevice


class MicrophoneInput(AudioInputDevice):
    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int = 24000,
        chunk_size: int = 4800,
    ):
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._audio: pyaudio.PyAudio | None = None
        self._stream = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    async def start(self) -> None:
        if self._active:
            return

        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sample_rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=self._chunk_size,
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

    async def stream_chunks(self) -> AsyncIterator[bytes]:
        """Yield audio chunks from microphone"""
        while self._active and self._stream:
            chunk = await asyncio.get_event_loop().run_in_executor(
                None, self._stream.read, self._chunk_size
            )
            yield chunk
