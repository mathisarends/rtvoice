import base64
import logging
from pathlib import Path

from transitbus import EventBus

from rtvoice.audio.audio_mixer import ConversationAudioMixer
from rtvoice.events.views import (
    AgentStoppedEvent,
    AssistantStartedRespondingEvent,
    AudioPlaybackCompletedEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioBufferAppendEvent,
    ResponseOutputAudioDeltaEvent,
)

logger = logging.getLogger(__name__)


class ConversationAudioRecorder:
    def __init__(self, event_bus: EventBus, output_path: Path):
        self._mixer = ConversationAudioMixer(output_path)
        self._assistant_speaking = False

        event_bus.on(AssistantStartedRespondingEvent, self._on_assistant_started)
        event_bus.on(AudioPlaybackCompletedEvent, self._on_assistant_stopped)
        event_bus.on(InputAudioBufferAppendEvent, self._on_user_audio)
        event_bus.on(ResponseOutputAudioDeltaEvent, self._on_assistant_audio)
        event_bus.on(AgentStoppedEvent, self._on_agent_stopped)

    async def _on_assistant_started(self, _: AssistantStartedRespondingEvent) -> None:
        self._assistant_speaking = True

    async def _on_assistant_stopped(self, _: AudioPlaybackCompletedEvent) -> None:
        self._assistant_speaking = False
        self._mixer.finalize()

    async def _on_user_audio(self, event: InputAudioBufferAppendEvent) -> None:
        if self._assistant_speaking:
            return
        self._mixer.feed_user(base64.b64decode(event.audio))

    async def _on_assistant_audio(self, event: ResponseOutputAudioDeltaEvent) -> None:
        self._mixer.feed_assistant(base64.b64decode(event.delta))

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        self._mixer.save()
        logger.info("Recording saved to %s", self._mixer.path)
