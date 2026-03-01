import base64
import logging
from pathlib import Path

from rtvoice.events import EventBus
from rtvoice.events.views import AgentStoppedEvent
from rtvoice.realtime.schemas import (
    InputAudioBufferAppendEvent,
    ResponseOutputAudioDeltaEvent,
)
from rtvoice.recording import AudioRecorder

logger = logging.getLogger(__name__)


class AudioRecordingWatchdog:
    def __init__(self, event_bus: EventBus, output_path: Path):
        self._recorder = AudioRecorder()
        self._output_path = output_path
        event_bus.subscribe(InputAudioBufferAppendEvent, self._on_user_audio)
        event_bus.subscribe(ResponseOutputAudioDeltaEvent, self._on_assistant_audio)
        event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)

    async def _on_user_audio(self, event: InputAudioBufferAppendEvent) -> None:
        self._recorder.record_user(base64.b64decode(event.audio))

    async def _on_assistant_audio(self, event: ResponseOutputAudioDeltaEvent) -> None:
        self._recorder.record_assistant(base64.b64decode(event.delta))

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        self._recorder.save(self._output_path)
        logger.info("Recording saved to %s", self._output_path)
