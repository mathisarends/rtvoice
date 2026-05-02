from collections.abc import AsyncIterator

from rtvoice.audio.ports import AudioInputDevice, AudioOutputDevice


class AudioSession:
    def __init__(
        self,
        input_device: AudioInputDevice,
        output_device: AudioOutputDevice,
    ):
        self._input = input_device
        self._output = output_device

    @property
    def is_playing(self) -> bool:
        return self._output.is_playing

    async def start(self) -> None:
        await self._input.start()
        await self._output.start()

    async def stop(self) -> None:
        await self._input.stop()
        await self._output.stop()

    def stream_input_chunks(self) -> AsyncIterator[bytes]:
        return self._input.stream_chunks()

    async def play_chunk(self, chunk: bytes) -> None:
        await self._output.play_chunk(chunk)

    async def clear_output_buffer(self) -> None:
        await self._output.clear_buffer()
