import base64

from rtvoice.audio.devices import AudioOutputDevice
from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentStartedEvent,
    AgentStoppedEvent,
    VolumeUpdateRequestedEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioBufferSpeechStartedEvent,
    ResponseOutputAudioDeltaEvent,
)
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
        self._event_bus.subscribe(
            InputAudioBufferSpeechStartedEvent,
            self._on_user_started_speaking,
        )
        self._event_bus.subscribe(
            VolumeUpdateRequestedEvent, self._on_volume_update_requested
        )

    async def _on_agent_started(self, _: AgentStartedEvent) -> None:
        await self._device.start()
        self.logger.info("Audio output started")

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        await self._device.stop()
        self.logger.info("Audio output stopped")

    async def _on_audio_delta(self, event: ResponseOutputAudioDeltaEvent) -> None:
        audio_bytes = base64.b64decode(event.delta)
        await self._device.play_chunk(audio_bytes)

    async def _on_user_started_speaking(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        self.logger.info("User started speaking - clearing audio output buffer")
        await self._device.clear_buffer()

    async def _on_volume_update_requested(
        self, event: VolumeUpdateRequestedEvent
    ) -> None:
        await self._device.set_volume(event.volume)
        percentage = int(event.volume * 100)
        self.logger.info("Audio output volume set to %d%%", percentage)
