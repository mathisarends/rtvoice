import base64

from rtvoice.audio.devices import AudioOutputDevice
from rtvoice.events import EventBus
from rtvoice.events.views import AgentStartedEvent, AgentStoppedEvent
from rtvoice.realtime.schemas import ResponseOutputAudioDeltaEvent
from rtvoice.shared.logging import LoggingMixin


class AudioOutputWatchdog(LoggingMixin):
    def __init__(
        self,
        event_bus: EventBus,
        device: AudioOutputDevice,
    ):
        self._event_bus = event_bus
        self._device = device

        self._event_bus.subscribe(AgentStartedEvent, self._on_agent_started)
        self._event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)
        self._event_bus.subscribe(ResponseOutputAudioDeltaEvent, self._on_audio_delta)

    async def _on_agent_started(self, _: AgentStartedEvent) -> None:
        await self._device.start()
        self.logger.info("Audio output started")

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        await self._device.stop()
        self.logger.info("Audio output stopped")

    async def _on_audio_delta(self, event: ResponseOutputAudioDeltaEvent) -> None:
        audio_bytes = base64.b64decode(event.delta)
        await self._device.write_chunk(audio_bytes)
