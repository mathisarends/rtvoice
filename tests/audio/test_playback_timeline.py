import pytest

from rtvoice.audio.echo import PlaybackTimeline

SAMPLE_RATE = 100


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def pcm(value: int, samples: int) -> bytes:
    return value.to_bytes(2, "little", signed=True) * samples


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def timeline(clock: FakeClock) -> PlaybackTimeline:
    return PlaybackTimeline(SAMPLE_RATE, history_seconds=2.0, clock=clock)


class TestRead:
    def test_empty_timeline_reads_silence(self, timeline: PlaybackTimeline) -> None:
        assert timeline.read(1000.0, 10) == pcm(0, 10)

    def test_written_chunk_reads_back_at_write_time(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        timeline.write(pcm(500, 10))

        assert timeline.read(clock.now, 10) == pcm(500, 10)

    def test_read_before_playback_is_silent(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        timeline.write(pcm(500, 10))

        assert timeline.read(clock.now - 0.1, 10) == pcm(0, 10)

    def test_read_spanning_playback_start_is_half_silent(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        timeline.write(pcm(500, 10))

        window = timeline.read(clock.now - 0.05, 10)

        assert window == pcm(0, 5) + pcm(500, 5)

    def test_read_past_end_of_playback_is_silent(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        timeline.write(pcm(500, 10))

        assert timeline.read(clock.now + 0.1, 10) == pcm(0, 10)


class TestScheduling:
    def test_backlogged_chunks_are_queued_back_to_back(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        start = clock.now
        timeline.write(pcm(100, 10))
        timeline.write(pcm(200, 10))

        assert timeline.read(start, 20) == pcm(100, 10) + pcm(200, 10)

    def test_chunk_after_drain_starts_at_write_time(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        timeline.write(pcm(100, 10))
        clock.advance(0.5)
        timeline.write(pcm(200, 10))

        assert timeline.read(clock.now, 10) == pcm(200, 10)

    def test_gap_between_chunks_reads_as_silence(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        timeline.write(pcm(100, 10))
        clock.advance(0.5)
        timeline.write(pcm(200, 10))

        assert timeline.read(clock.now - 0.2, 10) == pcm(0, 10)


class TestDiscardPending:
    def test_queued_audio_is_dropped(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        timeline.write(pcm(100, 100))  # one second of playback

        clock.advance(0.1)
        timeline.discard_pending()

        assert timeline.read(clock.now, 10) == pcm(0, 10)

    def test_already_played_audio_is_kept(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        start = clock.now
        timeline.write(pcm(100, 100))

        clock.advance(0.1)
        timeline.discard_pending()

        assert timeline.read(start, 10) == pcm(100, 10)

    def test_next_chunk_starts_at_write_time(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        timeline.write(pcm(100, 100))
        clock.advance(0.1)
        timeline.discard_pending()

        timeline.write(pcm(200, 10))

        assert timeline.read(clock.now, 10) == pcm(200, 10)


class TestHistory:
    def test_audio_beyond_history_is_forgotten(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        start = clock.now
        timeline.write(pcm(100, 10))

        clock.advance(3.0)
        timeline.write(pcm(200, 10))

        assert timeline.read(start, 10) == pcm(0, 10)

    def test_reset_clears_everything(
        self, timeline: PlaybackTimeline, clock: FakeClock
    ) -> None:
        timeline.write(pcm(100, 10))

        timeline.reset()

        assert timeline.read(clock.now, 10) == pcm(0, 10)
