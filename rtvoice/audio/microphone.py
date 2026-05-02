import asyncio
import threading
from collections.abc import AsyncIterator

from rtvoice.audio.devices import AudioInputDevice


class MicrophoneInput(AudioInputDevice):
    """Default microphone input powered by PyAudio.

    Streams raw 16-bit PCM audio from the system microphone in a
    non-blocking loop. Requires the `pyaudio` package — install it
    with `pip install rtvoice[audio]`.

    Example:
        ```python
        mic = MicrophoneInput(sample_rate=24000)
        agent = RealtimeAgent(audio_input=mic)
        ```
    """

    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int = 24000,
        chunk_size: int = 4800,
    ):
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._audio = None
        self._stream = None
        self._active = False
        self._read_complete = threading.Event()
        self._read_complete.set()

    @property
    def is_active(self) -> bool:
        return self._active

    async def start(self) -> None:
        if self._active:
            return

        try:
            import pyaudio
        except ImportError as e:
            raise ImportError(
                "pyaudio is required for MicrophoneInput. "
                "Install it with: pip install rtvoice[audio]"
            ) from e

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

    def _safe_read(self) -> bytes | None:
        self._read_complete.clear()
        try:
            if not self._active or not self._stream:
                return None
            return self._stream.read(self._chunk_size)
        except OSError:
            return None
        finally:
            self._read_complete.set()

    async def stop(self) -> None:
        if not self._active:
            return

        self._active = False

        await asyncio.get_event_loop().run_in_executor(
            None, self._read_complete.wait, 1.0
        )

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._audio:
            self._audio.terminate()
            self._audio = None

    async def stream_chunks(self) -> AsyncIterator[bytes]:
        while self._active and self._stream:
            chunk = await asyncio.get_event_loop().run_in_executor(
                None, self._safe_read
            )
            if chunk is None:
                break
            yield chunk
