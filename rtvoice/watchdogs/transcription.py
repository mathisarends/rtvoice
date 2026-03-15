import logging

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AssistantTranscriptChunkReceivedEvent,
    AssistantTranscriptCompletedEvent,
    AssistantTranscriptDeltaEvent,
    UserTranscriptCompletedEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioTranscriptionCompleted,
    ResponseOutputAudioTranscriptDone,
    ResponseOutputTextDelta,
    ResponseOutputTextDone,
    ResponseTextDelta,
    ResponseTextDone,
)

logger = logging.getLogger(__name__)


class TranscriptionWatchdog:
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus

        self._event_bus.subscribe(
            InputAudioTranscriptionCompleted,
            self._on_user_transcript_completed,
        )
        self._event_bus.subscribe(
            ResponseOutputAudioTranscriptDone,
            self._on_assistant_transcript_completed,
        )
        self._event_bus.subscribe(
            ResponseOutputTextDelta,
            self._on_assistant_text_delta,
        )
        self._event_bus.subscribe(
            ResponseTextDelta,
            self._on_assistant_text_delta,
        )
        self._event_bus.subscribe(
            ResponseOutputTextDone,
            self._on_assistant_text_done,
        )
        self._event_bus.subscribe(
            ResponseTextDone,
            self._on_assistant_text_done,
        )

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

        await self._event_bus.dispatch(
            UserTranscriptCompletedEvent(
                transcript=event.transcript, item_id=event.item_id
            )
        )

    async def _on_assistant_transcript_completed(
        self, event: ResponseOutputAudioTranscriptDone
    ) -> None:
        logger.info(
            "Assistant transcript completed: '%s' (response_id=%s)",
            event.transcript,
            event.response_id,
        )

        await self._event_bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript=event.transcript,
                item_id=event.item_id,
                output_index=event.output_index,
                content_index=event.content_index,
            )
        )

    async def _on_assistant_text_delta(
        self, event: ResponseOutputTextDelta | ResponseTextDelta
    ) -> None:
        await self._event_bus.dispatch(
            AssistantTranscriptChunkReceivedEvent(chunk=event.delta)
        )
        await self._event_bus.dispatch(
            AssistantTranscriptDeltaEvent(
                delta=event.delta,
                item_id=event.item_id,
                output_index=event.output_index,
                content_index=event.content_index,
            )
        )

    async def _on_assistant_text_done(
        self, event: ResponseOutputTextDone | ResponseTextDone
    ) -> None:
        await self._event_bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript=event.text,
                item_id=event.item_id,
                output_index=event.output_index,
                content_index=event.content_index,
            )
        )
