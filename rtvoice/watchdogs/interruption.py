import time

from rtvoice.audio.session import AudioSession
from rtvoice.events import EventBus
from rtvoice.events.views import AssistantInterruptedEvent
from rtvoice.realtime.schemas import (
    ConversationItemTruncateEvent,
    InputAudioBufferSpeechStartedEvent,
    OutputAudioBufferClearEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputAudioDeltaEvent,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.shared.logging import LoggingMixin


class InterruptionWatchdog(LoggingMixin):
    """Handles barge-in: cancels the running response, clears the audio buffer,
    and truncates the conversation item to what was actually played."""

    def __init__(
        self,
        event_bus: EventBus,
        websocket: RealtimeWebSocket,
        session: AudioSession,
    ):
        self._event_bus = event_bus
        self._websocket = websocket
        self._session = session

        self._response_id: str | None = None
        self._item_id: str | None = None
        self._start_time: float | None = None
        self._assistant_is_speaking = False

        self._event_bus.subscribe(ResponseCreatedEvent, self._on_response_created)
        self._event_bus.subscribe(ResponseOutputAudioDeltaEvent, self._on_audio_delta)
        self._event_bus.subscribe(ResponseDoneEvent, self._on_response_done)
        self._event_bus.subscribe(
            InputAudioBufferSpeechStartedEvent, self._on_user_started_speaking
        )

    @property
    def _elapsed_ms(self) -> int | None:
        if self._start_time is None:
            return None
        return int((time.time() - self._start_time) * 1000)

    async def _on_response_created(self, event: ResponseCreatedEvent) -> None:
        self._response_id = event.response_id
        self._start_time = time.time()
        self._assistant_is_speaking = True
        self.logger.debug("Response started: %s", event.response_id)

    async def _on_audio_delta(self, event: ResponseOutputAudioDeltaEvent) -> None:
        if event.response_id != self._response_id:
            return
        if not self._item_id:
            self._item_id = event.item_id
            self.logger.debug("Tracking item_id: %s", self._item_id)

    async def _on_response_done(self, event: ResponseDoneEvent) -> None:
        if event.response_id != self._response_id:
            return
        self.logger.debug("Response completed: %s", event.response_id)
        self._reset()

    async def _on_user_started_speaking(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        if not self._assistant_is_speaking and not self._session.is_playing:
            return

        self.logger.info("Barge-in detected - cancelling response")

        await self._websocket.send(ResponseCancelEvent())
        await self._websocket.send(OutputAudioBufferClearEvent())

        if self._item_id and self._elapsed_ms is not None:
            self.logger.debug(
                "Truncating item %s at %d ms", self._item_id, self._elapsed_ms
            )
            await self._websocket.send(
                ConversationItemTruncateEvent(
                    item_id=self._item_id,
                    content_index=0,
                    audio_end_ms=self._elapsed_ms,
                )
            )
        else:
            self.logger.warning(
                "Cannot truncate - missing item_id=%s or elapsed_ms=%s",
                self._item_id,
                self._elapsed_ms,
            )

        await self._event_bus.dispatch(
            AssistantInterruptedEvent(
                item_id=self._item_id,
                played_ms=self._elapsed_ms,
            )
        )

        self._reset()

    def _reset(self) -> None:
        self._response_id = None
        self._item_id = None
        self._start_time = None
        self._assistant_is_speaking = False
