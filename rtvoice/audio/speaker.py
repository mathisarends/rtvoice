import asyncio

import pyaudio

from rtvoice.audio.devices import AudioOutputDevice


class SpeakerOutput(AudioOutputDevice):
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

    async def write_chunk(self, chunk: bytes) -> None:
        if not self._active or not self._stream:
            return

        await asyncio.get_event_loop().run_in_executor(None, self._stream.write, chunk)
