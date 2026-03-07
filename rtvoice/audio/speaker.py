import logging
import queue
import threading
from typing import Annotated

from typing_extensions import Doc

from rtvoice.audio.devices import AudioOutputDevice

logger = logging.getLogger(__name__)


class SpeakerOutput(AudioOutputDevice):
    """Default speaker output powered by PyAudio.

    Plays raw 16-bit PCM audio through the system speaker using a
    dedicated background thread, keeping the async event loop unblocked.
    Requires the `pyaudio` package — install it with
    `pip install rtvoice[audio]`.

    Example:
        ```python
        speaker = SpeakerOutput(sample_rate=24000)
        agent = RealtimeAgent(audio_output=speaker)
        ```
    """

    def __init__(
        self,
        device_index: Annotated[
            int | None,
            Doc("PyAudio device index. Defaults to the system default output device."),
        ] = None,
        sample_rate: Annotated[
            int,
            Doc("Sample rate in Hz. Must match the model's output rate (24 000 Hz)."),
        ] = 24000,
    ):
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._audio = None
        self._stream = None
        self._active = False

        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._playback_thread: threading.Thread | None = None
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing or not self._queue.empty()

    async def start(self) -> None:
        if self._active:
            return

        try:
            import pyaudio
        except ImportError as e:
            raise ImportError(
                "pyaudio is required for SpeakerOutput. "
                "Install it with: pip install rtvoice[audio]"
            ) from e

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
                self._stream.write(chunk)
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
        logger.debug("Cleared %d audio chunks from queue", cleared)
