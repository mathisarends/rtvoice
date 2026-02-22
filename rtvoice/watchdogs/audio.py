import asyncio
import base64
from contextlib import suppress

from rtvoice.audio.session import AudioSession
from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentStartedEvent,
    AgentStoppedEvent,
    AudioPlaybackCompletedEvent,
    VolumeUpdateRequestedEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioBufferAppendEvent,
    InputAudioBufferSpeechStartedEvent,
    ResponseDoneEvent,
    ResponseOutputAudioDeltaEvent,
)
from rtvoice.shared.logging import LoggingMixin


class AudioWatchdog(LoggingMixin):
    def __init__(self, event_bus: EventBus, session: AudioSession):
        self._event_bus = event_bus
        self._session = session
        self._streaming_task: asyncio.Task | None = None

        self._event_bus.subscribe(AgentStartedEvent, self._on_agent_started)
        self._event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)
        self._event_bus.subscribe(ResponseOutputAudioDeltaEvent, self._on_audio_delta)
        self._event_bus.subscribe(
            InputAudioBufferSpeechStartedEvent, self._on_user_started_speaking
        )
        self._event_bus.subscribe(
            VolumeUpdateRequestedEvent, self._on_volume_update_requested
        )
        self._event_bus.subscribe(ResponseDoneEvent, self._on_response_done)

    async def _on_agent_started(self, _: AgentStartedEvent) -> None:
        await self._session.start()
        self._streaming_task = asyncio.create_task(self._stream_audio())
        self.logger.info("Audio started")

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        if self._streaming_task:
            self._streaming_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._streaming_task
            self._streaming_task = None

        await self._session.stop()
        self.logger.info("Audio stopped")

    async def _stream_audio(self) -> None:
        try:
            async for chunk in self._session.stream_input_chunks():
                base64_audio = base64.b64encode(chunk).decode("utf-8")
                await self._event_bus.dispatch(
                    InputAudioBufferAppendEvent(audio=base64_audio)
                )
        except asyncio.CancelledError:
            pass

    async def _on_audio_delta(self, event: ResponseOutputAudioDeltaEvent) -> None:
        audio_bytes = base64.b64decode(event.delta)
        await self._session.play_chunk(audio_bytes)

    async def _on_user_started_speaking(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        await self._session.clear_output_buffer()

    async def _on_volume_update_requested(
        self, event: VolumeUpdateRequestedEvent
    ) -> None:
        await self._session.set_volume(event.volume)
        self.logger.info("Volume set to %d%%", int(event.volume * 100))

    async def _on_response_done(self, _: ResponseDoneEvent) -> None:
        asyncio.create_task(self._wait_for_playback_completion())

    async def _wait_for_playback_completion(self) -> None:
        while self._session.is_playing:
            await asyncio.sleep(0.05)
        await self._event_bus.dispatch(AudioPlaybackCompletedEvent())
