import base64
import logging
from pathlib import Path

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentStoppedEvent,
    AssistantStartedRespondingEvent,
    AudioPlaybackCompletedEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioBufferAppendEvent,
    ResponseOutputAudioDeltaEvent,
)
from rtvoice.recording import AudioRecorder

logger = logging.getLogger(__name__)


class AudioRecordingWatchdog:
    def __init__(self, event_bus: EventBus, output_path: Path):
        self._recorder = AudioRecorder(output_path)
        self._assistant_speaking = False

        event_bus.subscribe(AssistantStartedRespondingEvent, self._on_assistant_started)
        event_bus.subscribe(AudioPlaybackCompletedEvent, self._on_assistant_stopped)
        event_bus.subscribe(InputAudioBufferAppendEvent, self._on_user_audio)
        event_bus.subscribe(ResponseOutputAudioDeltaEvent, self._on_assistant_audio)
        event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)

    async def _on_assistant_started(self, _: AssistantStartedRespondingEvent) -> None:
        self._assistant_speaking = True

    async def _on_assistant_stopped(self, _: AudioPlaybackCompletedEvent) -> None:
        self._assistant_speaking = False
        self._recorder.mark_end()

    async def _on_user_audio(self, event: InputAudioBufferAppendEvent) -> None:
        if self._assistant_speaking:
            return
        self._recorder.record_user(base64.b64decode(event.audio))

    async def _on_assistant_audio(self, event: ResponseOutputAudioDeltaEvent) -> None:
        self._recorder.record_assistant(base64.b64decode(event.delta))

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        self._recorder.save()
        logger.info("Recording saved to %s", self._recorder._path)
