import logging

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AssistantTranscriptChunkReceivedEvent,
    AssistantTranscriptCompletedEvent,
    UserTranscriptChunkReceivedEvent,
    UserTranscriptCompletedEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioTranscriptionCompleted,
    InputAudioTranscriptionDelta,
    ResponseOutputAudioTranscriptDelta,
    ResponseOutputAudioTranscriptDone,
)

logger = logging.getLogger(__name__)


class TranscriptionWatchdog:
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus

        self._event_bus.subscribe(
            InputAudioTranscriptionDelta,
            self._on_user_transcript_chunk,
        )
        self._event_bus.subscribe(
            InputAudioTranscriptionCompleted,
            self._on_user_transcript_completed,
        )

        self._event_bus.subscribe(
            ResponseOutputAudioTranscriptDelta,
            self._on_assistant_transcript_chunk,
        )
        self._event_bus.subscribe(
            ResponseOutputAudioTranscriptDone,
            self._on_assistant_transcript_completed,
        )

    async def _on_user_transcript_chunk(
        self, event: InputAudioTranscriptionDelta
    ) -> None:
        logger.debug(
            "User transcript chunk: '%s' (item_id=%s)",
            event.delta,
            event.item_id,
        )

        chunk_event = UserTranscriptChunkReceivedEvent(chunk=event.delta)
        await self._event_bus.dispatch(chunk_event)

    async def _on_user_transcript_completed(
        self, event: InputAudioTranscriptionCompleted
    ) -> None:
        logger.info(
            "User transcript completed: '%s' (item_id=%s)",
            event.transcript,
            event.item_id,
        )

        if event.usage:
            logger.debug("Transcription usage: %s", event.usage)

        completed_event = UserTranscriptCompletedEvent(
            transcript=event.transcript, item_id=event.item_id
        )
        await self._event_bus.dispatch(completed_event)

    async def _on_assistant_transcript_chunk(
        self, event: ResponseOutputAudioTranscriptDelta
    ) -> None:
        logger.debug(
            "Assistant transcript chunk: '%s' (response_id=%s)",
            event.delta,
            event.response_id,
        )

        chunk_event = AssistantTranscriptChunkReceivedEvent(chunk=event.delta)
        await self._event_bus.dispatch(chunk_event)

    async def _on_assistant_transcript_completed(
        self, event: ResponseOutputAudioTranscriptDone
    ) -> None:
        logger.info(
            "Assistant transcript completed: '%s' (response_id=%s)",
            event.transcript,
            event.response_id,
        )

        completed_event = AssistantTranscriptCompletedEvent(
            transcript=event.transcript,
            item_id=event.item_id,
            output_index=event.output_index,
            content_index=event.content_index,
        )
        await self._event_bus.dispatch(completed_event)
