import asyncio
import struct
import wave
from pathlib import Path


class ConversationAudioMixer:
    def __init__(self, path: str | Path, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._start_time: float | None = None
        self._user_chunks: list[tuple[float, bytes]] = []
        self._assistant_audio: bytearray = bytearray()
        self._assistant_start_time: float | None = None
        self._last_audio_time: float | None = None

    @property
    def path(self) -> Path:
        return self._path

    def _now(self) -> float:
        loop = asyncio.get_event_loop()
        if self._start_time is None:
            self._start_time = loop.time()
        return loop.time() - self._start_time

    def feed_user(self, data: bytes) -> None:
        self._user_chunks.append((self._now(), data))

    def feed_assistant(self, data: bytes) -> None:
        if self._assistant_start_time is None:
            self._assistant_start_time = self._now()
        self._assistant_audio.extend(data)

    def finalize(self) -> None:
        """Compute the final mixed timeline length for both tracks."""
        assistant_duration = len(self._assistant_audio) / 2 / self.sample_rate
        assistant_end = (self._assistant_start_time or 0) + assistant_duration
        self._last_audio_time = max(self._last_user_end(), assistant_end)

    def save(self) -> None:
        total_samples = int((self._last_audio_time or 0) * self.sample_rate)

        if total_samples == 0:
            return

        user_track = self._render_track(self._user_chunks, total_samples)

        assistant_offset_samples = int(
            (self._assistant_start_time or 0) * self.sample_rate
        )
        assistant_track = bytearray(total_samples * 2)
        offset_bytes = assistant_offset_samples * 2
        usable = self._assistant_audio[: total_samples * 2 - offset_bytes]
        if offset_bytes < total_samples * 2:
            assistant_track[offset_bytes : offset_bytes + len(usable)] = usable

        mono = bytearray()
        for i in range(total_samples):
            u = struct.unpack_from("<h", user_track, i * 2)[0]
            a = struct.unpack_from("<h", assistant_track, i * 2)[0]
            mixed = max(-32768, min(32767, u + a))
            mono += struct.pack("<h", mixed)

        with wave.open(str(self._path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.sample_rate)
            f.writeframes(bytes(mono))

    def _last_user_end(self) -> float:
        if not self._user_chunks:
            return 0.0
        ts, data = self._user_chunks[-1]
        return ts + len(data) / 2 / self.sample_rate

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
