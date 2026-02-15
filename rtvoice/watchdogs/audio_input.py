import asyncio
import base64
from contextlib import suppress

from rtvoice.audio.devices import AudioInputDevice
from rtvoice.events import EventBus
from rtvoice.events.views import AgentStartedEvent, AgentStoppedEvent
from rtvoice.realtime.schemas import InputAudioBufferAppendEvent
from rtvoice.shared.logging import LoggingMixin


class AudioInputWatchdog(LoggingMixin):
    def __init__(
        self,
        event_bus: EventBus,
        device: AudioInputDevice,
    ):
        self._event_bus = event_bus
        self._device = device
        self._streaming_task: asyncio.Task | None = None

        self._event_bus.subscribe(AgentStartedEvent, self._on_agent_started)
        self._event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)

    async def _on_agent_started(self, _: AgentStartedEvent) -> None:
        await self._device.start()
        self._streaming_task = asyncio.create_task(self._stream_audio())
        self.logger.info("Audio input started")

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        if self._streaming_task:
            self._streaming_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._streaming_task
            self._streaming_task = None

        await self._device.stop()
        self.logger.info("Audio input stopped")

    async def _stream_audio(self) -> None:
        try:
            async for chunk in self._device.stream_chunks():
                base64_audio = base64.b64encode(chunk).decode("utf-8")
                event = InputAudioBufferAppendEvent(audio=base64_audio)
                await self._event_bus.dispatch(event)
        except asyncio.CancelledError:
            pass
