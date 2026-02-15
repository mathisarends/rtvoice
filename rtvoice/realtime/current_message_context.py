from __future__ import annotations

import time

from rtvoice.events.schemas import ResponseOutputAudioDeltaEvent

from rtvoice.events import EventBus
from rtvoice.shared.logging import LoggingMixin
from rtvoice.state.base import VoiceAssistantEvent


class CurrentMessageContext(LoggingMixin):
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._start_time: float | None = None
        self._item_id: str | None = None

        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_STARTED_RESPONDING, self._on_response_started
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED, self._on_response_ended
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED, self._on_response_ended
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.AUDIO_CHUNK_RECEIVED, self._handle_audio_chunk_received
        )

        self.logger.info("CurrentMessageContext initialized and subscribed to events")

    @property
    def item_id(self) -> str | None:
        return self._item_id

    @property
    def current_duration_ms(self) -> int | None:
        if self._start_time is None:
            return None
        return int((time.time() - self._start_time) * 1000)

    async def _on_response_started(self) -> None:
        self._start_time = time.time()
        self.logger.debug("Assistant response started - timer started")

    async def _on_response_ended(self, event: VoiceAssistantEvent) -> None:
        self._start_time = None
        self._item_id = None

        if event == VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED:
            self.logger.debug(
                "Assistant response completed normally - Resetting CurrentMessageContext"
            )
        elif event == VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED:
            self.logger.debug(
                "Assistant speech interrupted by user - Resetting CurrentMessageContext"
            )
        else:
            self.logger.debug(
                "Assistant response ended (unknown reason) - Resetting CurrentMessageContext"
            )

    def _handle_audio_chunk_received(
        self, response_output_audio_delta: ResponseOutputAudioDeltaEvent
    ) -> None:
        """Audio chunk received -> AUDIO_CHUNK_RECEIVED"""
        if self._item_id:
            return

        self._item_id = response_output_audio_delta.item_id
        self.logger.debug("Set item_id to: %s", self._item_id)
