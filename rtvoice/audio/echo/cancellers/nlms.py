import math

import numpy as np

from rtvoice.audio.echo.ports import EchoCanceller

_INT16_PEAK = 32768.0
_EPS = 1e-12
_FAR_END_FLOOR = 1e-3  # Below -60 dBFS there is no useful echo to learn.


class NlmsEchoCanceller(EchoCanceller):
    """NLMS filter with residual suppression for nonlinear speaker echo."""

    def __init__(
        self,
        sample_rate: int = 24000,
        *,
        tail_ms: float = 250.0,
        block_size: int = 256,
        step_size: float = 0.4,
        residual_leakage: float = 0.1,
    ):
        self._block = block_size
        self._fft = 2 * block_size
        self._bins = block_size + 1
        self._partitions = max(1, math.ceil(tail_ms / 1000 * sample_rate / block_size))
        self._step_size = step_size
        self._residual_leakage = residual_leakage

        self._weights = np.zeros((self._partitions, self._bins), dtype=np.complex128)
        self._far_spectra = np.zeros_like(self._weights)
        self._far_tail = np.zeros(self._block)
        self._power = np.full(self._bins, _EPS)
        self._gain = np.ones(self._bins)
        self._pad = np.zeros(self._block)
        self._cursor = 0

        self._near_pending = np.zeros(0)
        self._far_pending = np.zeros(0)

    def process(self, near_end: bytes, far_end: bytes) -> bytes:
        self._near_pending = np.concatenate([self._near_pending, _decode(near_end)])
        self._far_pending = np.concatenate([self._far_pending, _decode(far_end)])

        blocks = []
        while len(self._near_pending) >= self._block:
            near, self._near_pending = self._split(self._near_pending)
            far, self._far_pending = self._split(self._far_pending)
            blocks.append(self._process_block(near, far))

        if not blocks:
            return b""
        return _encode(np.concatenate(blocks))

    def reset(self) -> None:
        self._weights.fill(0)
        self._far_spectra.fill(0)
        self._far_tail.fill(0)
        self._power.fill(_EPS)
        self._gain.fill(1.0)
        self._cursor = 0
        self._near_pending = np.zeros(0)
        self._far_pending = np.zeros(0)

    def _split(self, pending: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pending[: self._block], pending[self._block :]

    def _process_block(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(np.concatenate([self._far_tail, far]))
        self._far_tail = far
        self._far_spectra[self._cursor] = spectrum
        history = self._far_spectra[
            (self._cursor - np.arange(self._partitions)) % self._partitions
        ]
        self._cursor = (self._cursor + 1) % self._partitions

        echo_spectrum = np.sum(self._weights * history, axis=0)
        echo = np.fft.irfft(echo_spectrum, n=self._fft)[self._block :]
        residual = near - echo
        residual_spectrum = np.fft.rfft(np.concatenate([self._pad, residual]))

        if np.max(np.abs(far)) < _FAR_END_FLOOR:
            self._gain = 0.5 * self._gain + 0.5
            return residual

        self._adapt(history, spectrum, residual_spectrum, near, residual)
        return self._suppress(echo_spectrum, residual_spectrum, residual)

    def _adapt(
        self,
        history: np.ndarray,
        spectrum: np.ndarray,
        residual_spectrum: np.ndarray,
        near: np.ndarray,
        residual: np.ndarray,
    ) -> None:
        # Pull back a diverging filter before it amplifies the feedback loop.
        if residual @ residual > 4 * (near @ near):
            self._weights *= 0.5
            return

        self._power = 0.8 * self._power + 0.2 * np.abs(spectrum) ** 2
        step = self._step_size / (self._partitions * self._power + _EPS)
        self._weights += self._constrain(np.conj(history) * (step * residual_spectrum))

    def _constrain(self, gradient: np.ndarray) -> np.ndarray:
        # Circular wrap-around is not a real tap and would make the filter noncausal.
        taps = np.fft.irfft(gradient, n=self._fft, axis=-1)
        taps[:, self._block :] = 0.0
        return np.fft.rfft(taps, axis=-1)

    def _suppress(
        self,
        echo_spectrum: np.ndarray,
        residual_spectrum: np.ndarray,
        residual: np.ndarray,
    ) -> np.ndarray:
        if self._residual_leakage <= 0:
            return residual

        residual_power = np.abs(residual_spectrum) ** 2
        echo_power = np.abs(echo_spectrum) ** 2
        target = residual_power / (
            residual_power + self._residual_leakage * echo_power + _EPS
        )
        # Smoothing avoids musical noise from abrupt gain changes.
        self._gain = 0.5 * self._gain + 0.5 * target
        return np.fft.irfft(residual_spectrum * self._gain, n=self._fft)[self._block :]


def _decode(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / _INT16_PEAK


def _encode(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples * _INT16_PEAK, -_INT16_PEAK, _INT16_PEAK - 1)
    return clipped.astype("<i2").tobytes()
