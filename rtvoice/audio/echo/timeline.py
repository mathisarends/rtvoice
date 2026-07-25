import time

from rtvoice.audio.echo.ports import Clock

_BYTES_PER_SAMPLE = 2


class PlaybackTimeline:
    """Reconstructs timing because output devices expose order, not timestamps."""

    def __init__(
        self,
        sample_rate: int = 24000,
        *,
        history_seconds: float = 2.0,
        clock: Clock = time.monotonic,
    ):
        self._sample_rate = sample_rate
        self._history_seconds = history_seconds
        self._clock = clock
        self._buffer = bytearray()
        self._origin: float | None = None
        self._cursor = 0.0

    def reset(self) -> None:
        self._buffer.clear()
        self._origin = None
        self._cursor = 0.0

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return

        now = self._clock()
        start = now if self._origin is None else max(self._cursor, now)

        if self._origin is None:
            self._origin = start

        self._put(start, chunk)
        self._cursor = start + self._duration(chunk)
        self._trim(now)

    def discard_pending(self) -> None:
        if self._origin is None:
            return

        now = self._clock()
        del self._buffer[max(self._offset(now), 0) * _BYTES_PER_SAMPLE :]
        self._cursor = now

    def read(self, start: float, num_samples: int) -> bytes:
        out = bytearray(num_samples * _BYTES_PER_SAMPLE)

        if self._origin is None:
            return bytes(out)

        offset = self._offset(start)
        src = max(offset, 0)
        dst = max(-offset, 0)
        count = min(num_samples - dst, len(self._buffer) // _BYTES_PER_SAMPLE - src)

        if count > 0:
            out[dst * _BYTES_PER_SAMPLE : (dst + count) * _BYTES_PER_SAMPLE] = (
                self._buffer[
                    src * _BYTES_PER_SAMPLE : (src + count) * _BYTES_PER_SAMPLE
                ]
            )

        return bytes(out)

    def _put(self, start: float, chunk: bytes) -> None:
        offset = self._offset(start)

        if offset < 0:
            chunk = chunk[-offset * _BYTES_PER_SAMPLE :]
            offset = 0
            if not chunk:
                return

        end = offset * _BYTES_PER_SAMPLE + len(chunk)
        if end > len(self._buffer):
            self._buffer.extend(bytes(end - len(self._buffer)))

        self._buffer[offset * _BYTES_PER_SAMPLE : end] = chunk

    def _trim(self, now: float) -> None:
        dropped = min(
            self._offset(now - self._history_seconds),
            len(self._buffer) // _BYTES_PER_SAMPLE,
        )
        if dropped <= 0:
            return

        del self._buffer[: dropped * _BYTES_PER_SAMPLE]

        if self._buffer:
            self._origin = (self._origin or 0.0) + dropped / self._sample_rate
        else:
            self._origin = None

    def _offset(self, at: float) -> int:
        return round((at - (self._origin or 0.0)) * self._sample_rate)

    def _duration(self, chunk: bytes) -> float:
        return len(chunk) / _BYTES_PER_SAMPLE / self._sample_rate
