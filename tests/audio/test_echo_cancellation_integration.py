from collections.abc import AsyncIterator

import pytest

np = pytest.importorskip("numpy")

from rtvoice.audio.echo import EchoCancellation
from rtvoice.audio.ports import AudioInput, AudioOutput

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, samples: int) -> None:
        self.now += samples / SAMPLE_RATE


class RoomInput(AudioInput):
    def __init__(self, chunks: list[bytes], clock: FakeClock):
        self._chunks = chunks
        self._clock = clock
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    async def start(self) -> None:
        self._active = True

    async def stop(self) -> None:
        self._active = False

    async def stream_chunks(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            self._clock.advance(len(chunk) // 2)
            yield chunk


class SpeakerOutput(AudioOutput):
    def __init__(self) -> None:
        self.played: list[bytes] = []

    @property
    def is_playing(self) -> bool:
        return bool(self.played)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def play_chunk(self, chunk: bytes) -> None:
        self.played.append(chunk)

    async def clear_buffer(self) -> None:
        self.played.clear()


def encode(samples: np.ndarray) -> bytes:
    return np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes()


def decode(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768


def chunks(samples: np.ndarray) -> list[bytes]:
    return [
        encode(samples[start : start + FRAME_SAMPLES])
        for start in range(0, len(samples), FRAME_SAMPLES)
    ]


def energy(samples: np.ndarray) -> float:
    return float(samples @ samples) or 1e-12


@pytest.mark.asyncio
async def test_wrapped_devices_cancel_simulated_room_echo() -> None:
    rng = np.random.default_rng(7)
    playback_signal = rng.normal(0, 0.2, SAMPLE_RATE * 4)
    room_path = np.zeros(160)
    room_path[120] = 0.5
    room_path[150] = -0.2
    captured_echo = np.convolve(playback_signal, room_path)[: len(playback_signal)]

    clock = FakeClock()
    speaker = SpeakerOutput()
    capture, playback = EchoCancellation(
        sample_rate=SAMPLE_RATE,
        alignment_margin_s=0.0,
        clock=clock,
    ).wrap(RoomInput(chunks(captured_echo), clock), speaker)

    await playback.start()
    await capture.start()
    for chunk in chunks(playback_signal):
        await playback.play_chunk(chunk)

    residual = np.concatenate(
        [decode(chunk) async for chunk in capture.stream_chunks()]
    )

    tail = len(residual) // 4
    attenuation_db = 10 * np.log10(
        energy(residual[-tail:]) / energy(captured_echo[-tail:])
    )
    assert attenuation_db < -20
    assert b"".join(speaker.played) == encode(playback_signal)
