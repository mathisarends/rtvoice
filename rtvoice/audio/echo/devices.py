import time
from collections.abc import AsyncIterator

from rtvoice.audio.echo.ports import Clock, EchoCanceller
from rtvoice.audio.echo.timeline import PlaybackTimeline
from rtvoice.audio.ports import AudioInputDevice, AudioOutputDevice

_BYTES_PER_SAMPLE = 2


class ReferenceTapOutput(AudioOutputDevice):
    """Decorator that mirrors everything handed to the wrapped device into the shared
    timeline, so the capture side knows what the speaker is about to emit."""

    def __init__(self, output: AudioOutputDevice, timeline: PlaybackTimeline):
        self._output = output
        self._timeline = timeline

    @property
    def is_playing(self) -> bool:
        return self._output.is_playing

    async def start(self) -> None:
        self._timeline.reset()
        await self._output.start()

    async def stop(self) -> None:
        await self._output.stop()
        self._timeline.reset()

    async def play_chunk(self, chunk: bytes) -> None:
        self._timeline.write(chunk)
        await self._output.play_chunk(chunk)

    async def clear_buffer(self) -> None:
        await self._output.clear_buffer()
        self._timeline.discard_pending()


class EchoCancellingInput(AudioInputDevice):
    """Decorator that subtracts the assistant's own playback from the captured stream
    before anyone - VAD included - gets to see it."""

    def __init__(
        self,
        input_device: AudioInputDevice,
        timeline: PlaybackTimeline,
        canceller: EchoCanceller,
        *,
        sample_rate: int = 24000,
        alignment_margin_s: float = 0.04,
        resync_threshold_s: float = 0.1,
        clock: Clock = time.monotonic,
    ):
        self._input = input_device
        self._timeline = timeline
        self._canceller = canceller
        self._sample_rate = sample_rate
        self._alignment_margin_s = alignment_margin_s
        self._resync_threshold_s = resync_threshold_s
        self._clock = clock
        self._capture_cursor: float | None = None

    @property
    def is_active(self) -> bool:
        return self._input.is_active

    async def start(self) -> None:
        self._capture_cursor = None
        self._canceller.reset()
        await self._input.start()

    async def stop(self) -> None:
        await self._input.stop()

    async def stream_chunks(self) -> AsyncIterator[bytes]:
        async for chunk in self._input.stream_chunks():
            cleaned = self._canceller.process(chunk, self._reference_for(chunk))
            if cleaned:
                yield cleaned

    def _reference_for(self, chunk: bytes) -> bytes:
        samples = len(chunk) // _BYTES_PER_SAMPLE
        duration = samples / self._sample_rate
        start = self._capture_start(self._clock() - duration, duration)

        # deliberately read a little early: capture latency makes the true instant no
        # newer than our estimate, and a canceller can only look backwards in time
        return self._timeline.read(start - self._alignment_margin_s, samples)

    def _capture_start(self, estimate: float, duration: float) -> float:
        cursor = self._capture_cursor

        # advance in sample time so the far-end reads stay gap-free under scheduling
        # jitter, and re-anchor to the wall clock only on real drift
        if cursor is None or abs(cursor - estimate) > self._resync_threshold_s:
            cursor = estimate

        self._capture_cursor = cursor + duration
        return cursor
