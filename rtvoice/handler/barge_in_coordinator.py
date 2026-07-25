import logging
import time
from collections.abc import Callable

from transitbus import EventBus

from rtvoice.audio.session import AudioSession
from rtvoice.events.views import (
    AssistantInterruptedEvent,
    AudioPlaybackCompletedEvent,
    InterruptAssistantCommand,
)
from rtvoice.realtime.schemas import (
    ConversationItemTruncateEvent,
    InputAudioBufferSpeechStartedEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputAudioDeltaEvent,
)
from rtvoice.realtime.websocket import RealtimeWebSocket

logger = logging.getLogger(__name__)


class BargeInCoordinator:
    """Handles interruptions - user barge-in as well as programmatic ones: cancels
    the running response, clears the audio buffer, and truncates the conversation
    item to what was actually played."""

    def __init__(
        self,
        event_bus: EventBus,
        websocket: RealtimeWebSocket,
        audio_session: AudioSession,
        speech_speed: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._event_bus = event_bus
        self._websocket = websocket
        self._audio_session = audio_session
        self._speech_speed = speech_speed
        self._clock = clock

        self._response_id: str | None = None
        self._item_id: str | None = None
        self._start_time: float | None = None
        self._assistant_is_speaking = False

        self._event_bus.on(ResponseCreatedEvent, self._on_response_created)
        self._event_bus.on(ResponseOutputAudioDeltaEvent, self._on_audio_delta)
        self._event_bus.on(ResponseDoneEvent, self._on_response_done)
        self._event_bus.on(AudioPlaybackCompletedEvent, self._on_playback_completed)
        self._event_bus.on(
            InputAudioBufferSpeechStartedEvent, self._on_user_started_speaking
        )
        self._event_bus.on(InterruptAssistantCommand, self._on_interrupt_requested)

    @property
    def _elapsed_ms(self) -> int | None:
        if self._start_time is None:
            return None
        return int((self._clock() - self._start_time) * 1000)

    async def _on_response_created(self, event: ResponseCreatedEvent) -> None:
        self._response_id = event.response_id
        self._start_time = None
        self._assistant_is_speaking = True
        logger.debug("Response started: %s", event.response_id)

    async def _on_audio_delta(self, event: ResponseOutputAudioDeltaEvent) -> None:
        if event.response_id != self._response_id:
            return
        if not self._item_id:
            self._item_id = event.item_id
            self._start_time = self._clock()
            logger.debug("Tracking item_id: %s", self._item_id)

    async def _on_response_done(self, event: ResponseDoneEvent) -> None:
        if event.response_id != self._response_id:
            return
        logger.debug("Response completed: %s", event.response_id)
        self._assistant_is_speaking = False
        if self._item_id is None:
            self._reset()

    async def _on_playback_completed(self, _: AudioPlaybackCompletedEvent) -> None:
        if not self._assistant_is_speaking:
            self._reset()

    async def _on_user_started_speaking(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        await self._interrupt("Barge-in detected")

    async def _on_interrupt_requested(self, _: InterruptAssistantCommand) -> None:
        await self._interrupt("Interrupt requested")

    async def _interrupt(self, cause: str) -> None:
        if (
            not self._assistant_is_speaking
            and self._item_id is None
            and not self._audio_session.is_playing
        ):
            return

        logger.info("%s - cancelling response", cause)

        if self._assistant_is_speaking:
            await self._websocket.send(ResponseCancelEvent())

        played_ms = self._elapsed_ms
        if self._item_id and played_ms is not None:
            logger.debug("Truncating item %s at %d ms", self._item_id, played_ms)
            await self._websocket.send(
                ConversationItemTruncateEvent(
                    item_id=self._item_id,
                    content_index=0,
                    audio_end_ms=played_ms,
                )
            )
        else:
            logger.warning(
                "Cannot truncate - missing item_id=%s or elapsed_ms=%s",
                self._item_id,
                played_ms,
            )

        await self._event_bus.dispatch(
            AssistantInterruptedEvent(
                item_id=self._item_id,
                played_ms=played_ms,
                response_id=self._response_id,
                speech_speed=self._speech_speed,
            )
        )

        self._reset()

    def _reset(self) -> None:
        self._response_id = None
        self._item_id = None
        self._start_time = None
        self._assistant_is_speaking = False

    def set_speech_speed(self, speed: float) -> None:
        self._speech_speed = speed
