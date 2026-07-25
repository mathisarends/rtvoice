import time

from rtvoice.audio.echo.cancellers import NlmsEchoCanceller
from rtvoice.audio.echo.devices import EchoCancellingInput, ReferenceTapOutput
from rtvoice.audio.echo.ports import Clock, EchoCanceller
from rtvoice.audio.echo.timeline import PlaybackTimeline
from rtvoice.audio.ports import AudioInputDevice, AudioOutputDevice


class EchoCancellation:
    def __init__(
        self,
        canceller: EchoCanceller | None = None,
        *,
        sample_rate: int = 24000,
        alignment_margin_s: float = 0.04,
        history_seconds: float = 2.0,
        clock: Clock = time.monotonic,
    ):
        self._canceller = canceller or NlmsEchoCanceller(sample_rate)
        self._sample_rate = sample_rate
        self._alignment_margin_s = alignment_margin_s
        self._history_seconds = history_seconds
        self._clock = clock
        self._wrapped = False

    def wrap(
        self, input_device: AudioInputDevice, output_device: AudioOutputDevice
    ) -> tuple[AudioInputDevice, AudioOutputDevice]:
        if self._wrapped:
            raise RuntimeError(
                "EchoCancellation holds filter state for a single device pair - "
                "create a new instance per pair."
            )
        self._wrapped = True

        timeline = PlaybackTimeline(
            self._sample_rate,
            history_seconds=self._history_seconds,
            clock=self._clock,
        )
        return (
            EchoCancellingInput(
                input_device,
                timeline,
                self._canceller,
                sample_rate=self._sample_rate,
                alignment_margin_s=self._alignment_margin_s,
                clock=self._clock,
            ),
            ReferenceTapOutput(output_device, timeline),
        )
