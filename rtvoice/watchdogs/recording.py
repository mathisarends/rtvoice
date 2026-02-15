import base64

from rtvoice.events import EventBus
from rtvoice.events.views import AgentStartedEvent, AgentStoppedEvent
from rtvoice.realtime.schemas import (
    InputAudioBufferAppendEvent,
    ResponseOutputAudioDeltaEvent,
)
from rtvoice.recording.service import ConversationRecorder
from rtvoice.shared.logging import LoggingMixin


class RecordingWatchdog(LoggingMixin):
    def __init__(self, event_bus: EventBus, output_path: str):
        self._event_bus = event_bus
        self._recorder = ConversationRecorder(output_path)

        self._event_bus.subscribe(AgentStartedEvent, self._on_agent_started)
        self._event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)
        self._event_bus.subscribe(
            InputAudioBufferAppendEvent, self._on_input_audio_append
        )
        self._event_bus.subscribe(
            ResponseOutputAudioDeltaEvent, self._on_response_audio_delta
        )

    async def _on_agent_started(self, event: AgentStartedEvent) -> None:
        await self._recorder.start_recording()
        self.logger.info("Recording started")

    async def _on_agent_stopped(self, event: AgentStoppedEvent) -> None:
        await self._recorder.stop_recording()
        self.logger.info("Recording stopped")

    async def _on_input_audio_append(self, event: InputAudioBufferAppendEvent) -> None:
        audio_bytes = base64.b64decode(event.audio)
        await self._recorder.write_audio_chunk(audio_bytes)

    async def _on_response_audio_delta(
        self, event: ResponseOutputAudioDeltaEvent
    ) -> None:
        audio_bytes = base64.b64decode(event.delta)
        await self._recorder.write_audio_chunk(audio_bytes)
