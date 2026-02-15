from rtvoice.events import EventBus
from rtvoice.realtime.schemas import (
    InputAudioTranscriptionCompleted,
    ResponseOutputAudioTranscriptDone,
)
from rtvoice.shared.logging import LoggingMixin
from rtvoice.state.base import VoiceAssistantEvent


class TranscriptionWatchdog(LoggingMixin):
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus

        self._event_bus.subscribe(
            VoiceAssistantEvent.USER_TRANSCRIPT_COMPLETED,
            self._on_user_transcript_completed,
        )
        self._event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_TRANSCRIPT_COMPLETED,
            self._on_assistant_transcript_completed,
        )

    async def _on_user_transcript_completed(
        self, data: InputAudioTranscriptionCompleted
    ) -> None:
        self.logger.info(
            "User transcript completed: '%s' (item_id=%s)",
            data.transcript,
            data.item_id,
        )

        if data.usage:
            self.logger.debug("Transcription usage: %s", data.usage)

    async def _on_assistant_transcript_completed(
        self, data: ResponseOutputAudioTranscriptDone
    ) -> None:
        self.logger.info(
            "Assistant transcript completed: '%s' (response_id=%s)",
            data.transcript,
            data.response_id,
        )
