import logging

from transitbus import EventBus

from rtvoice.events.views import (
    AssistantTranscriptCompletedEvent,
    UserTranscriptCompletedEvent,
)

logger = logging.getLogger(__name__)


class TranscriptLogger:
    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.on(UserTranscriptCompletedEvent, self._on_user)
        self._event_bus.on(AssistantTranscriptCompletedEvent, self._on_assistant)

    async def _on_user(self, event: UserTranscriptCompletedEvent) -> None:
        logger.info("[user] %s", event.transcript)

    async def _on_assistant(self, event: AssistantTranscriptCompletedEvent) -> None:
        logger.info("[assistant] %s", event.transcript)
