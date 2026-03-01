import logging

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AssistantStartedRespondingEvent,
    AssistantStoppedRespondingEvent,
    AudioPlaybackCompletedEvent,
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ResponseCreatedEvent,
)

logger = logging.getLogger(__name__)


class SpeechStateWatchdog:
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus

        self._event_bus.subscribe(
            InputAudioBufferSpeechStartedEvent, self._on_speech_started
        )
        self._event_bus.subscribe(
            InputAudioBufferSpeechStoppedEvent, self._on_speech_stopped
        )
        self._event_bus.subscribe(ResponseCreatedEvent, self._on_response_created)
        self._event_bus.subscribe(
            AudioPlaybackCompletedEvent, self._on_playback_completed
        )

    async def _on_speech_started(self, _: InputAudioBufferSpeechStartedEvent) -> None:
        await self._event_bus.dispatch(UserStartedSpeakingEvent())

    async def _on_speech_stopped(self, _: InputAudioBufferSpeechStoppedEvent) -> None:
        await self._event_bus.dispatch(UserStoppedSpeakingEvent())

    async def _on_response_created(self, _: ResponseCreatedEvent) -> None:
        await self._event_bus.dispatch(AssistantStartedRespondingEvent())

    async def _on_playback_completed(self, _: AudioPlaybackCompletedEvent) -> None:
        await self._event_bus.dispatch(AssistantStoppedRespondingEvent())
