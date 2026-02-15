from rtvoice.events import EventBus
from rtvoice.realtime.schemas import (
    InputAudioBufferSpeechStartedEvent,
    OutputAudioBufferClearEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.shared.logging import LoggingMixin


class InterruptionWatchdog(LoggingMixin):
    def __init__(self, event_bus: EventBus, websocket: RealtimeWebSocket):
        self._event_bus = event_bus
        self._websocket = websocket
        self._assistant_is_speaking = False

        self._event_bus.subscribe(ResponseCreatedEvent, self._on_response_created)
        self._event_bus.subscribe(ResponseDoneEvent, self._on_response_done)
        self._event_bus.subscribe(
            InputAudioBufferSpeechStartedEvent,
            self._on_user_started_speaking,
        )

    async def _on_response_created(self, _: ResponseCreatedEvent) -> None:
        self._assistant_is_speaking = True
        self.logger.debug("Assistant started speaking")

    async def _on_response_done(self, _: ResponseDoneEvent) -> None:
        self._assistant_is_speaking = False
        self.logger.debug("Assistant finished speaking")

    async def _on_user_started_speaking(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        if not self._assistant_is_speaking:
            return

        self.logger.info("User interrupted assistant - cancelling response")

        await self._websocket.send(ResponseCancelEvent())
        await self._websocket.send(OutputAudioBufferClearEvent())

        self._assistant_is_speaking = False
