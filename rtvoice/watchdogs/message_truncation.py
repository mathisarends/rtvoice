import time

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AssistantResponseCompletedEvent,
    AssistantSpeechInterruptedEvent,
    AssistantStartedRespondingEvent,
    AudioChunkReceivedEvent,
    MessageTruncationRequestedEvent,
)
from rtvoice.shared.logging import LoggingMixin


class MessageTruncationWatchdog(LoggingMixin):
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus

        self._start_time: float | None = None
        self._item_id: str | None = None

        self._event_bus.subscribe(
            AssistantStartedRespondingEvent, self._on_response_started
        )
        self._event_bus.subscribe(
            AssistantResponseCompletedEvent, self._on_response_completed
        )
        self._event_bus.subscribe(
            AssistantSpeechInterruptedEvent, self._on_speech_interrupted
        )
        self._event_bus.subscribe(
            AudioChunkReceivedEvent, self._on_audio_chunk_received
        )

    @property
    def current_duration_ms(self) -> int | None:
        if self._start_time is None:
            return None
        return int((time.time() - self._start_time) * 1000)

    async def _on_response_started(self, _: AssistantStartedRespondingEvent) -> None:
        self._start_time = time.time()
        self.logger.debug("Assistant response started - timer started")

    async def _on_response_completed(self, _: AssistantResponseCompletedEvent) -> None:
        self.logger.debug("Assistant response completed - resetting state")
        self._reset_state()

    async def _on_speech_interrupted(
        self, event: AssistantSpeechInterruptedEvent
    ) -> None:
        if not self._item_id or self.current_duration_ms is None:
            self.logger.warning("Cannot truncate - missing item_id or duration")
            self._reset_state()
            return

        self.logger.info(
            "Publishing truncation request for item %s at %d ms",
            self._item_id,
            self.current_duration_ms,
        )

        truncation_event = MessageTruncationRequestedEvent(
            item_id=self._item_id,
            audio_end_ms=self.current_duration_ms,
        )

        await self._event_bus.dispatch(truncation_event)
        self.logger.debug("Truncation request published successfully")

        self._reset_state()

    async def _on_audio_chunk_received(self, event: AudioChunkReceivedEvent) -> None:
        if self._item_id:
            return

        self._item_id = event.item_id
        self.logger.debug("Set item_id to: %s", self._item_id)

    def _reset_state(self) -> None:
        self._start_time = None
        self._item_id = None
