import pytest

np = pytest.importorskip("numpy")

from rtvoice.audio.echo import NlmsEchoCanceller

SAMPLE_RATE = 16000
ECHO_DELAY_SAMPLES = 120


def far_end_signal(samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.2, samples)


def echo_of(far: np.ndarray) -> np.ndarray:
    path = np.zeros(ECHO_DELAY_SAMPLES + 40)
    path[ECHO_DELAY_SAMPLES] = 0.5
    path[ECHO_DELAY_SAMPLES + 30] = -0.2
    return np.convolve(far, path)[: len(far)]


def encode(samples: np.ndarray) -> bytes:
    return np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes()


def decode(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768


def energy(samples: np.ndarray) -> float:
    return float(samples @ samples) or 1e-12


def run(canceller: NlmsEchoCanceller, near: np.ndarray, far: np.ndarray) -> np.ndarray:
    frame = 512
    out = [
        decode(
            canceller.process(encode(near[i : i + frame]), encode(far[i : i + frame]))
        )
        for i in range(0, len(near), frame)
    ]
    return np.concatenate(out)


@pytest.fixture
def canceller() -> NlmsEchoCanceller:
    return NlmsEchoCanceller(SAMPLE_RATE, tail_ms=30.0, block_size=64)


class TestConvergence:
    def test_removes_the_echo_when_only_the_speaker_is_active(
        self, canceller: NlmsEchoCanceller
    ) -> None:
        far = far_end_signal(SAMPLE_RATE * 4)
        near = echo_of(far)

        residual = run(canceller, near, far)

        tail = len(residual) // 4
        attenuation_db = 10 * np.log10(energy(residual[-tail:]) / energy(near[-tail:]))
        assert attenuation_db < -20

    def test_leaves_the_microphone_untouched_without_playback(
        self, canceller: NlmsEchoCanceller
    ) -> None:
        near = far_end_signal(SAMPLE_RATE, seed=7)
        far = np.zeros_like(near)

        residual = run(canceller, near, far)

        assert np.allclose(residual, near[: len(residual)], atol=1e-4)

    def test_keeps_near_end_speech_while_the_assistant_talks(
        self, canceller: NlmsEchoCanceller
    ) -> None:
        far = far_end_signal(SAMPLE_RATE * 4)
        run(canceller, echo_of(far), far)  # converge on echo only

        far = far_end_signal(SAMPLE_RATE, seed=1)
        speech = far_end_signal(SAMPLE_RATE, seed=3)
        residual = run(canceller, echo_of(far) + speech, far)

        kept_db = 10 * np.log10(energy(residual) / energy(speech[: len(residual)]))
        assert kept_db > -6


class TestStreamContract:
    def test_output_tracks_input_length_within_one_block(
        self, canceller: NlmsEchoCanceller
    ) -> None:
        far = far_end_signal(SAMPLE_RATE)

        residual = run(canceller, echo_of(far), far)

        assert 0 <= len(far) - len(residual) < 64

    def test_partial_block_is_buffered_until_complete(self) -> None:
        canceller = NlmsEchoCanceller(SAMPLE_RATE, block_size=64)
        silence = encode(np.zeros(32))

        assert canceller.process(silence, silence) == b""
        assert len(canceller.process(silence, silence)) == 64 * 2

    def test_reset_drops_adaptation(self, canceller: NlmsEchoCanceller) -> None:
        far = far_end_signal(SAMPLE_RATE * 4)
        run(canceller, echo_of(far), far)

        canceller.reset()
        residual = run(canceller, echo_of(far)[:512], far[:512])

        assert energy(residual) > 0
