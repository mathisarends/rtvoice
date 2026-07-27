import asyncio
import logging

from transitbus import EventBus

from rtvoice.conversation.inactivity_timer import ConversationInactivityTimer
from rtvoice.events.views import (
    AgentSessionConnectedEvent,
    AudioPlaybackCompletedEvent,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    UserInactivityCountdownEvent,
    UserInactivityTimeoutEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ResponseCreatedEvent,
)

logger = logging.getLogger(__name__)

_COUNTDOWN_SECONDS = frozenset({5, 4, 3, 2, 1})


class ConversationInactivityMonitor:
    def __init__(self, event_bus: EventBus, timeout_seconds: float = 10.0):
        self.event_bus = event_bus
        self._timer = ConversationInactivityTimer(timeout_seconds)
        self._is_monitoring = False
        self._check_task: asyncio.Task | None = None
        self._assistant_is_speaking = False
        self._assistant_response_pending = False
        self._user_has_stopped_speaking = False
        self._active_tool_responses: set[str] = set()
        self._monitoring_generation = 0
        self.event_bus.on(AgentSessionConnectedEvent, self._handle_session_connected)
        self.event_bus.on(
            InputAudioBufferSpeechStoppedEvent, self._handle_user_speech_ended
        )
        self.event_bus.on(
            InputAudioBufferSpeechStartedEvent, self._handle_user_started_speaking
        )
        self.event_bus.on(ResponseCreatedEvent, self._handle_assistant_started)
        self.event_bus.on(AudioPlaybackCompletedEvent, self._handle_assistant_done)
        self.event_bus.on(ToolExecutionStartedEvent, self._handle_tool_started)
        self.event_bus.on(ToolExecutionCompletedEvent, self._handle_tool_completed)

    async def _handle_session_connected(self, _: AgentSessionConnectedEvent) -> None:
        self._user_has_stopped_speaking = True
        logger.debug("Session connected - starting inactivity timeout monitoring")
        self._try_start_monitoring()

    async def _handle_user_speech_ended(
        self, event: InputAudioBufferSpeechStoppedEvent
    ) -> None:
        self._user_has_stopped_speaking = True
        logger.debug("User stopped speaking at %d ms", event.audio_end_ms)
        self._try_start_monitoring()

    async def _handle_user_started_speaking(
        self, event: InputAudioBufferSpeechStartedEvent
    ) -> None:
        self._user_has_stopped_speaking = False
        self._stop_monitoring()
        logger.debug(
            "User started speaking at %d ms, stopping inactivity timeout monitoring",
            event.audio_start_ms,
        )

    async def _handle_assistant_started(self, _: ResponseCreatedEvent) -> None:
        self._assistant_response_pending = False
        self._assistant_is_speaking = True
        self._stop_monitoring()
        logger.debug("Assistant started speaking")

    async def _handle_assistant_done(self, _: AudioPlaybackCompletedEvent) -> None:
        if self._assistant_response_pending:
            return
        self._assistant_is_speaking = False
        logger.debug("Assistant finished speaking")
        self._try_start_monitoring()

    async def _handle_tool_started(self, event: ToolExecutionStartedEvent) -> None:
        self._active_tool_responses.add(event.response_id)
        self._stop_monitoring()
        logger.debug("Tool execution started [response_id=%s]", event.response_id)

    async def _handle_tool_completed(self, event: ToolExecutionCompletedEvent) -> None:
        self._active_tool_responses.discard(event.response_id)
        if event.response_pending:
            self._assistant_response_pending = True
            self._assistant_is_speaking = True
            self._stop_monitoring()
        else:
            if not self._assistant_response_pending:
                self._assistant_is_speaking = False
            self._try_start_monitoring()
        logger.debug(
            "Tool execution completed [response_id=%s, response_pending=%s]",
            event.response_id,
            event.response_pending,
        )

    def _try_start_monitoring(self) -> None:
        if (
            not self._user_has_stopped_speaking
            or self._assistant_is_speaking
            or self._assistant_response_pending
            or self._active_tool_responses
        ):
            return

        self._timer.reset()
        self._is_monitoring = True
        self._monitoring_generation += 1
        generation = self._monitoring_generation
        logger.debug(
            "Both user and assistant finished - starting inactivity timeout monitoring (%.1fs)",
            self._timer._timeout_seconds,
        )

        self._check_task = asyncio.create_task(self._monitor_timeout(generation))

    def _stop_monitoring(self) -> None:
        self._is_monitoring = False
        self._monitoring_generation += 1

    async def _monitor_timeout(self, generation: int | None = None) -> None:
        generation = generation or self._monitoring_generation
        dispatched_countdowns: set[int] = set()

        while self._is_monitoring and generation == self._monitoring_generation:
            if self._timer.has_timed_out():
                logger.warning(
                    "Inactivity timeout occurred after %.1f seconds",
                    self._timer._timeout_seconds,
                )
                await self.event_bus.dispatch(
                    UserInactivityTimeoutEvent(
                        timeout_seconds=self._timer._timeout_seconds
                    )
                )
                self._is_monitoring = False
                self._user_has_stopped_speaking = False
                break

            remaining = self._timer.remaining()
            if (
                remaining in _COUNTDOWN_SECONDS
                and remaining not in dispatched_countdowns
            ):
                dispatched_countdowns.add(remaining)
                logger.debug("Inactivity countdown: %d seconds remaining", remaining)
                await self.event_bus.dispatch(
                    UserInactivityCountdownEvent(remaining_seconds=remaining)
                )

            await asyncio.sleep(0.25)
