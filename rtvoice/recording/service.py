import asyncio
import struct
import wave
from pathlib import Path


class AudioRecorder:
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._start_time: float | None = None
        self._user_chunks: list[tuple[float, bytes]] = []
        self._assistant_chunks: list[tuple[float, bytes]] = []

    def _now(self) -> float:
        loop = asyncio.get_event_loop()
        if self._start_time is None:
            self._start_time = loop.time()
        return loop.time() - self._start_time

    def record_user(self, data: bytes) -> None:
        self._user_chunks.append((self._now(), data))

    def record_assistant(self, data: bytes) -> None:
        self._assistant_chunks.append((self._now(), data))

    def save(self, path: str | Path) -> None:
        total_samples = self._total_samples()
        user_track = self._render_track(self._user_chunks, total_samples)
        assistant_track = self._render_track(self._assistant_chunks, total_samples)

        stereo = bytearray()
        for i in range(total_samples):
            l_sample = struct.unpack_from("<h", user_track, i * 2)[0]
            r_sample = struct.unpack_from("<h", assistant_track, i * 2)[0]
            stereo += struct.pack("<hh", l_sample, r_sample)

        with wave.open(str(path), "wb") as f:
            f.setnchannels(2)
            f.setsampwidth(2)
            f.setframerate(self.sample_rate)
            f.writeframes(bytes(stereo))

    def _total_samples(self) -> int:
        def last_sample(chunks: list[tuple[float, bytes]]) -> int:
            if not chunks:
                return 0
            ts, data = chunks[-1]
            return int(ts * self.sample_rate) + len(data) // 2

        return max(last_sample(self._user_chunks), last_sample(self._assistant_chunks))

    def _render_track(
        self, chunks: list[tuple[float, bytes]], total_samples: int
    ) -> bytearray:
        buf = bytearray(total_samples * 2)
        for ts, data in chunks:
            offset_bytes = int(ts * self.sample_rate) * 2
            end = offset_bytes + len(data)
            if end <= len(buf):
                buf[offset_bytes:end] = data
        return buf
