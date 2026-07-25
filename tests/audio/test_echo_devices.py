from collections.abc import AsyncIterator

import pytest

from rtvoice.audio.echo import (
    EchoCancellation,
    EchoCanceller,
    EchoCancellingInput,
    PlaybackTimeline,
    ReferenceTapOutput,
)
from rtvoice.audio.ports import AudioInputDevice, AudioOutputDevice

SAMPLE_RATE = 100


def pcm(value: int, samples: int) -> bytes:
    return value.to_bytes(2, "little", signed=True) * samples


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeInput(AudioInputDevice):
    def __init__(self, chunks: list[bytes], clock: FakeClock | None = None):
        self._chunks = chunks
        self._clock = clock
        self.started = False
        self.stopped = False

    @property
    def is_active(self) -> bool:
        return self.started and not self.stopped

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def stream_chunks(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            if self._clock:
                self._clock.advance(len(chunk) // 2 / SAMPLE_RATE)
            yield chunk


class FakeOutput(AudioOutputDevice):
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.started = False
        self.stopped = False
        self.cleared = 0

    @property
    def is_playing(self) -> bool:
        return bool(self.played)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def play_chunk(self, chunk: bytes) -> None:
        self.played.append(chunk)

    async def clear_buffer(self) -> None:
        self.cleared += 1


class RecordingCanceller(EchoCanceller):
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, bytes]] = []
        self.resets = 0

    def process(self, near_end: bytes, far_end: bytes) -> bytes:
        self.calls.append((near_end, far_end))
        return near_end

    def reset(self) -> None:
        self.resets += 1


class SilencingCanceller(EchoCanceller):
    def process(self, near_end: bytes, far_end: bytes) -> bytes:
        return b"" if any(far_end) else near_end


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def timeline(clock: FakeClock) -> PlaybackTimeline:
    return PlaybackTimeline(SAMPLE_RATE, clock=clock)


class TestReferenceTapOutput:
    @pytest.mark.asyncio
    async def test_played_audio_reaches_the_timeline(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        output = ReferenceTapOutput(FakeOutput(), timeline)

        await output.play_chunk(pcm(700, 10))

        assert timeline.read(clock.now, 10) == pcm(700, 10)

    @pytest.mark.asyncio
    async def test_playback_is_forwarded_unchanged(
        self, timeline: PlaybackTimeline
    ) -> None:
        inner = FakeOutput()
        output = ReferenceTapOutput(inner, timeline)

        await output.play_chunk(pcm(700, 10))

        assert inner.played == [pcm(700, 10)]

    @pytest.mark.asyncio
    async def test_clearing_the_device_also_drops_the_reference(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        inner = FakeOutput()
        output = ReferenceTapOutput(inner, timeline)
        await output.play_chunk(pcm(700, 100))

        await output.clear_buffer()

        assert inner.cleared == 1
        assert timeline.read(clock.now, 10) == pcm(0, 10)

    @pytest.mark.asyncio
    async def test_lifecycle_is_delegated(self, timeline: PlaybackTimeline) -> None:
        inner = FakeOutput()
        output = ReferenceTapOutput(inner, timeline)

        await output.start()
        await output.stop()

        assert (inner.started, inner.stopped) == (True, True)

    def test_is_playing_is_delegated(self, timeline: PlaybackTimeline) -> None:
        inner = FakeOutput()

        assert ReferenceTapOutput(inner, timeline).is_playing is False


class TestEchoCancellingInput:
    @pytest.mark.asyncio
    async def test_capture_is_paired_with_the_matching_playback(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        canceller = RecordingCanceller()
        source = FakeInput([pcm(100, 10)], clock)
        capture = EchoCancellingInput(
            source,
            timeline,
            canceller,
            sample_rate=SAMPLE_RATE,
            alignment_margin_s=0.0,
            clock=clock,
        )
        timeline.write(pcm(700, 10))

        [chunk async for chunk in capture.stream_chunks()]

        assert canceller.calls == [(pcm(100, 10), pcm(700, 10))]

    @pytest.mark.asyncio
    async def test_reference_is_read_ahead_by_the_alignment_margin(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        canceller = RecordingCanceller()
        source = FakeInput([pcm(100, 10)], clock)
        capture = EchoCancellingInput(
            source,
            timeline,
            canceller,
            sample_rate=SAMPLE_RATE,
            alignment_margin_s=0.05,
            clock=clock,
        )
        timeline.write(pcm(700, 10))

        [chunk async for chunk in capture.stream_chunks()]

        _, reference = canceller.calls[0]
        assert reference == pcm(0, 5) + pcm(700, 5)

    @pytest.mark.asyncio
    async def test_silence_from_the_speaker_leaves_capture_untouched(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        source = FakeInput([pcm(100, 10), pcm(200, 10)], clock)
        capture = EchoCancellingInput(
            source, timeline, SilencingCanceller(), sample_rate=SAMPLE_RATE, clock=clock
        )

        chunks = [chunk async for chunk in capture.stream_chunks()]

        assert chunks == [pcm(100, 10), pcm(200, 10)]

    @pytest.mark.asyncio
    async def test_fully_cancelled_frames_are_not_forwarded(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        source = FakeInput([pcm(100, 10)], clock)
        capture = EchoCancellingInput(
            source,
            timeline,
            SilencingCanceller(),
            sample_rate=SAMPLE_RATE,
            alignment_margin_s=0.0,
            clock=clock,
        )
        timeline.write(pcm(700, 10))

        chunks = [chunk async for chunk in capture.stream_chunks()]

        assert chunks == []

    @pytest.mark.asyncio
    async def test_start_resets_the_canceller(self, timeline: PlaybackTimeline) -> None:
        canceller = RecordingCanceller()
        source = FakeInput([])
        capture = EchoCancellingInput(
            source, timeline, canceller, sample_rate=SAMPLE_RATE
        )

        await capture.start()

        assert canceller.resets == 1
        assert source.started is True

    @pytest.mark.asyncio
    async def test_lifecycle_is_delegated(self, timeline: PlaybackTimeline) -> None:
        source = FakeInput([])
        capture = EchoCancellingInput(
            source, timeline, RecordingCanceller(), sample_rate=SAMPLE_RATE
        )

        await capture.start()
        assert capture.is_active is True

        await capture.stop()
        assert capture.is_active is False


class TestEchoCancellation:
    @pytest.mark.asyncio
    async def test_wrapped_pair_shares_one_timeline(self, clock: FakeClock) -> None:
        canceller = RecordingCanceller()
        source = FakeInput([pcm(100, 10)], clock)
        capture, playback = EchoCancellation(
            canceller,
            sample_rate=SAMPLE_RATE,
            alignment_margin_s=0.0,
            clock=clock,
        ).wrap(source, FakeOutput())

        await playback.play_chunk(pcm(700, 10))
        [chunk async for chunk in capture.stream_chunks()]

        assert canceller.calls == [(pcm(100, 10), pcm(700, 10))]

    def test_reusing_an_instance_for_a_second_pair_is_rejected(self) -> None:
        echo_cancellation = EchoCancellation(RecordingCanceller())
        echo_cancellation.wrap(FakeInput([]), FakeOutput())

        with pytest.raises(RuntimeError, match="single device pair"):
            echo_cancellation.wrap(FakeInput([]), FakeOutput())
