import base64
from pathlib import Path

from rtvoice.events import EventBus
from rtvoice.events.views import AgentStartedEvent, AgentStoppedEvent
from rtvoice.realtime.schemas import (
    AudioFormat,
    InputAudioBufferAppendEvent,
    ResponseOutputAudioDeltaEvent,
)
from rtvoice.recording import AudioRecorder
from rtvoice.shared.logging import LoggingMixin


class RecordingWatchdog(LoggingMixin):
    def __init__(
        self,
        event_bus: EventBus,
        output_path: str | None,
        audio_format: AudioFormat = AudioFormat.PCM16,
    ):
        self._event_bus = event_bus
        self._output_path = output_path
        self._audio_format = audio_format
        self._recorder: AudioRecorder | None = None
        self._recording_path: str | None = None

        self._event_bus.subscribe(AgentStartedEvent, self._on_agent_started)
        self._event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)
        self._event_bus.subscribe(
            InputAudioBufferAppendEvent, self._on_input_audio_append
        )
        self._event_bus.subscribe(
            ResponseOutputAudioDeltaEvent, self._on_response_audio_delta
        )

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None

    @property
    def recording_path(self) -> str | None:
        return self._recording_path

    async def _on_agent_started(self, _: AgentStartedEvent) -> None:
        if not self._output_path:
            return

        output_file = Path(self._output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        self._recorder = AudioRecorder(
            output_file=self._output_path,
            audio_format=self._audio_format,
        )
        await self._recorder.start()
        self.logger.info("Recording started to %s", self._output_path)

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        if not self.is_recording:
            return

        self._recording_path = await self._recorder.stop()
        self.logger.info("Recording stopped at %s", self._recording_path)
        self._recorder = None

    async def _on_input_audio_append(self, event: InputAudioBufferAppendEvent) -> None:
        if not self.is_recording:
            return

        audio_bytes = base64.b64decode(event.audio)
        await self._recorder.write_chunk(audio_bytes)

    async def _on_response_audio_delta(
        self, event: ResponseOutputAudioDeltaEvent
    ) -> None:
        if not self.is_recording:
            return

        audio_bytes = base64.b64decode(event.delta)
        await self._recorder.write_chunk(audio_bytes)
