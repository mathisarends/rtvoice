import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.events.views import (
    AgentStoppedEvent,
    AssistantStartedRespondingEvent,
    AudioPlaybackCompletedEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioBufferAppendEvent,
    RealtimeServerEvent,
    ResponseOutputAudioDeltaEvent,
)
from rtvoice.watchdogs import AudioRecordingWatchdog


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def mock_recorder() -> MagicMock:
    recorder = MagicMock()
    recorder.mark_end = MagicMock()
    recorder.record_user = MagicMock()
    recorder.record_assistant = MagicMock()
    recorder.save = MagicMock()
    return recorder


@pytest.fixture
def watchdog(
    event_bus: EventBus, mock_recorder: MagicMock, tmp_path: Path
) -> AudioRecordingWatchdog:
    with patch("rtvoice.watchdogs.recording.AudioRecorder") as mock_cls:
        mock_cls.return_value = mock_recorder
        wd = AudioRecordingWatchdog(event_bus, tmp_path / "recording.wav")
    return wd


class TestAssistantSpeakingState:
    @pytest.mark.asyncio
    async def test_assistant_started_sets_speaking_flag(
        self, event_bus: EventBus, watchdog: AudioRecordingWatchdog
    ) -> None:
        await event_bus.dispatch(AssistantStartedRespondingEvent())

        assert watchdog._assistant_speaking is True

    @pytest.mark.asyncio
    async def test_playback_completed_clears_speaking_flag(
        self, event_bus: EventBus, watchdog: AudioRecordingWatchdog
    ) -> None:
        await event_bus.dispatch(AssistantStartedRespondingEvent())
        await event_bus.dispatch(AudioPlaybackCompletedEvent())

        assert watchdog._assistant_speaking is False

    @pytest.mark.asyncio
    async def test_playback_completed_calls_mark_end(
        self,
        event_bus: EventBus,
        watchdog: AudioRecordingWatchdog,
        mock_recorder: MagicMock,
    ) -> None:
        await event_bus.dispatch(AudioPlaybackCompletedEvent())

        mock_recorder.mark_end.assert_called_once()


class TestUserAudioRecording:
    @pytest.mark.asyncio
    async def test_user_audio_recorded_when_assistant_not_speaking(
        self,
        event_bus: EventBus,
        watchdog: AudioRecordingWatchdog,
        mock_recorder: MagicMock,
    ) -> None:
        audio_bytes = b"\x00\x01\x02"
        encoded = base64.b64encode(audio_bytes).decode()

        await event_bus.dispatch(InputAudioBufferAppendEvent(audio=encoded))

        mock_recorder.record_user.assert_called_once_with(audio_bytes)

    @pytest.mark.asyncio
    async def test_user_audio_skipped_when_assistant_speaking(
        self,
        event_bus: EventBus,
        watchdog: AudioRecordingWatchdog,
        mock_recorder: MagicMock,
    ) -> None:
        audio_bytes = b"\x00\x01\x02"
        encoded = base64.b64encode(audio_bytes).decode()

        await event_bus.dispatch(AssistantStartedRespondingEvent())
        await event_bus.dispatch(InputAudioBufferAppendEvent(audio=encoded))

        mock_recorder.record_user.assert_not_called()

    @pytest.mark.asyncio
    async def test_user_audio_recorded_again_after_playback_completed(
        self,
        event_bus: EventBus,
        watchdog: AudioRecordingWatchdog,
        mock_recorder: MagicMock,
    ) -> None:
        audio_bytes = b"\xaa\xbb"
        encoded = base64.b64encode(audio_bytes).decode()

        await event_bus.dispatch(AssistantStartedRespondingEvent())
        await event_bus.dispatch(AudioPlaybackCompletedEvent())
        await event_bus.dispatch(InputAudioBufferAppendEvent(audio=encoded))

        mock_recorder.record_user.assert_called_once_with(audio_bytes)


class TestAssistantAudioRecording:
    @pytest.mark.asyncio
    async def test_assistant_audio_delta_is_recorded(
        self,
        event_bus: EventBus,
        watchdog: AudioRecordingWatchdog,
        mock_recorder: MagicMock,
    ) -> None:
        audio_bytes = b"\x10\x20\x30"
        encoded = base64.b64encode(audio_bytes).decode()

        await event_bus.dispatch(
            ResponseOutputAudioDeltaEvent(
                type=RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DELTA,
                event_id="evt_001",
                item_id="item_001",
                response_id="resp_001",
                output_index=0,
                content_index=0,
                delta=encoded,
            )
        )

        mock_recorder.record_assistant.assert_called_once_with(audio_bytes)


class TestAgentStopped:
    @pytest.mark.asyncio
    async def test_agent_stopped_saves_recording(
        self,
        event_bus: EventBus,
        watchdog: AudioRecordingWatchdog,
        mock_recorder: MagicMock,
    ) -> None:
        await event_bus.dispatch(AgentStoppedEvent())

        mock_recorder.save.assert_called_once()
