import struct
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from rtvoice.audio.audio_mixer import ConversationAudioMixer


def pcm_bytes(num_samples: int, value: int = 0) -> bytes:
    return struct.pack(f"<{num_samples}h", *([value] * num_samples))


def read_wav_samples(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as f:
        raw = f.readframes(f.getnframes())
    return list(struct.unpack(f"<{len(raw) // 2}h", raw))


@pytest.fixture
def mixer(tmp_path: Path) -> ConversationAudioMixer:
    return ConversationAudioMixer(path=tmp_path / "out.wav", sample_rate=24000)


class TestInit:
    def test_creates_output_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "out.wav"
        ConversationAudioMixer(path=path)

        assert path.parent.exists()

    def test_default_sample_rate(self, tmp_path: Path) -> None:
        mixer = ConversationAudioMixer(path=tmp_path / "out.wav")

        assert mixer.sample_rate == 24000

    def test_custom_sample_rate(self, tmp_path: Path) -> None:
        mixer = ConversationAudioMixer(path=tmp_path / "out.wav", sample_rate=16000)

        assert mixer.sample_rate == 16000


class TestRecordUser:
    def test_stores_chunk_with_timestamp(self, mixer: ConversationAudioMixer) -> None:
        with patch.object(mixer, "_now", return_value=1.0):
            mixer.feed_user(pcm_bytes(100))

        assert len(mixer._user_chunks) == 1
        ts, _data = mixer._user_chunks[0]
        assert ts == 1.0

    def test_accumulates_multiple_chunks(self, mixer: ConversationAudioMixer) -> None:
        with patch.object(mixer, "_now", return_value=0.0):
            mixer.feed_user(pcm_bytes(100))
        with patch.object(mixer, "_now", return_value=0.5):
            mixer.feed_user(pcm_bytes(100))

        assert len(mixer._user_chunks) == 2


class TestRecordAssistant:
    def test_stores_audio_data(self, mixer: ConversationAudioMixer) -> None:
        data = pcm_bytes(100, value=1000)
        with patch.object(mixer, "_now", return_value=0.0):
            mixer.feed_assistant(data)

        assert mixer._assistant_audio == bytearray(data)

    def test_captures_start_time_on_first_chunk(
        self, mixer: ConversationAudioMixer
    ) -> None:
        with patch.object(mixer, "_now", return_value=2.5):
            mixer.feed_assistant(pcm_bytes(100))

        assert mixer._assistant_start_time == 2.5

    def test_does_not_overwrite_start_time_on_subsequent_chunks(
        self, mixer: ConversationAudioMixer
    ) -> None:
        with patch.object(mixer, "_now", return_value=1.0):
            mixer.feed_assistant(pcm_bytes(100))
        with patch.object(mixer, "_now", return_value=2.0):
            mixer.feed_assistant(pcm_bytes(100))

        assert mixer._assistant_start_time == 1.0

    def test_concatenates_chunks(self, mixer: ConversationAudioMixer) -> None:
        with patch.object(mixer, "_now", return_value=0.0):
            mixer.feed_assistant(pcm_bytes(50, value=100))
            mixer.feed_assistant(pcm_bytes(50, value=200))

        assert len(mixer._assistant_audio) == 200


class TestFinalize:
    def test_sets_last_audio_time(self, mixer: ConversationAudioMixer) -> None:
        with patch.object(mixer, "_now", return_value=0.0):
            mixer.feed_assistant(pcm_bytes(24000))

        mixer.finalize()

        assert mixer._last_audio_time is not None
        assert mixer._last_audio_time > 0

    def test_uses_user_end_when_later_than_assistant(
        self, mixer: ConversationAudioMixer
    ) -> None:
        with patch.object(mixer, "_now", return_value=0.0):
            mixer.feed_user(pcm_bytes(24000 * 5))
        with patch.object(mixer, "_now", return_value=0.0):
            mixer.feed_assistant(pcm_bytes(24000))

        mixer.finalize()

        assert mixer._last_audio_time == pytest.approx(5.0, abs=0.01)

    def test_handles_no_audio(self, mixer: ConversationAudioMixer) -> None:
        mixer.finalize()

        assert mixer._last_audio_time == 0.0


class TestSave:
    def test_writes_wav_file(
        self, mixer: ConversationAudioMixer, tmp_path: Path
    ) -> None:
        with patch.object(mixer, "_now", return_value=0.0):
            mixer.feed_user(pcm_bytes(24000))
        mixer.finalize()
        mixer.save()

        assert (tmp_path / "out.wav").exists()

    def test_wav_has_correct_sample_rate(
        self, mixer: ConversationAudioMixer, tmp_path: Path
    ) -> None:
        with patch.object(mixer, "_now", return_value=0.0):
            mixer.feed_user(pcm_bytes(24000))
        mixer.finalize()
        mixer.save()

        with wave.open(str(tmp_path / "out.wav"), "rb") as f:
            assert f.getframerate() == 24000

    def test_wav_is_mono(self, mixer: ConversationAudioMixer, tmp_path: Path) -> None:
        with patch.object(mixer, "_now", return_value=0.0):
            mixer.feed_user(pcm_bytes(24000))
        mixer.finalize()
        mixer.save()

        with wave.open(str(tmp_path / "out.wav"), "rb") as f:
            assert f.getnchannels() == 1

    def test_does_not_write_file_when_no_audio(
        self, mixer: ConversationAudioMixer, tmp_path: Path
    ) -> None:
        mixer.save()

        assert not (tmp_path / "out.wav").exists()

    def test_mixes_user_and_assistant_audio(
        self, mixer: ConversationAudioMixer, tmp_path: Path
    ) -> None:
        with patch.object(mixer, "_now", return_value=0.0):
            mixer.feed_user(pcm_bytes(24000, value=1000))
            mixer.feed_assistant(pcm_bytes(24000, value=2000))
        mixer.finalize()
        mixer.save()

        samples = read_wav_samples(tmp_path / "out.wav")
        assert any(s == 3000 for s in samples)

    def test_clamps_mixed_audio_to_int16_range(
        self, mixer: ConversationAudioMixer, tmp_path: Path
    ) -> None:
        with patch.object(mixer, "_now", return_value=0.0):
            mixer.feed_user(pcm_bytes(24000, value=30000))
            mixer.feed_assistant(pcm_bytes(24000, value=30000))
        mixer.finalize()
        mixer.save()

        samples = read_wav_samples(tmp_path / "out.wav")
        assert all(-32768 <= s <= 32767 for s in samples)

    def test_assistant_audio_placed_at_correct_offset(
        self, mixer: ConversationAudioMixer, tmp_path: Path
    ) -> None:
        assistant_chunk = pcm_bytes(24000, value=500)

        with patch.object(mixer, "_now", return_value=0.0):
            mixer.feed_user(pcm_bytes(24000 * 2, value=0))
        with patch.object(mixer, "_now", return_value=1.0):
            mixer.feed_assistant(assistant_chunk)

        mixer.finalize()
        mixer.save()

        samples = read_wav_samples(tmp_path / "out.wav")
        assert samples[0] == 0
        assert samples[24000] == 500


class TestRenderTrack:
    def test_places_chunk_at_correct_offset(
        self, mixer: ConversationAudioMixer
    ) -> None:
        chunk = pcm_bytes(100, value=999)
        result = mixer._render_track([(1.0, chunk)], total_samples=24100)

        offset = int(1.0 * 24000) * 2
        placed = struct.unpack_from("<h", result, offset)[0]
        assert placed == 999

    def test_ignores_chunks_that_exceed_buffer(
        self, mixer: ConversationAudioMixer
    ) -> None:
        chunk = pcm_bytes(1000, value=1)
        result = mixer._render_track([(0.99, chunk)], total_samples=100)

        assert len(result) == 200

    def test_empty_chunks_returns_silence(self, mixer: ConversationAudioMixer) -> None:
        result = mixer._render_track([], total_samples=100)

        assert result == bytearray(200)
