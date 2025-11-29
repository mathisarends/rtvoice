# rtvoice/state/listening.py
import asyncio

from rtvoice.state.base import AssistantState
from rtvoice.state.context import VoiceAssistantContext
from rtvoice.state.events import VoiceAssistantEvent
from rtvoice.state.models import StateType


class ListeningState(AssistantState):
    def __init__(self, timeout_seconds: float = 10.0):
        super().__init__()
        self._timeout_seconds = timeout_seconds
        self._timeout_task: asyncio.Task | None = None
        self._event_handlers = {
            VoiceAssistantEvent.USER_SPEECH_ENDED: self._handle_speech_ended,
            VoiceAssistantEvent.TIMEOUT_OCCURRED: self._handle_timeout,
        }

    @property
    def state_type(self) -> StateType:
        return StateType.LISTENING

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Entering Listening state - user is speaking")
        context.audio_player.clear_queue_and_stop_chunks()

        await self._state_machine.ensure_realtime_audio_channel_connected()

        self.logger.debug("Initiating realtime session for user conversation")
        await self._state_machine.start_realtime_session()

        await self._start_timeout(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._stop_timeout()

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        handler = self._event_handlers.get(event)
        if handler:
            await handler(context)

    async def _handle_speech_ended(self, context: VoiceAssistantContext) -> None:
        self.logger.info("User finished speaking")
        await self._transition_to_responding()

    async def _handle_timeout(self, context: VoiceAssistantContext) -> None:
        self.logger.warning(
            "Listening timeout - no speech detected within %s seconds, returning to idle",
            self._timeout_seconds,
        )
        await self._transition_to_idle()

    async def _start_timeout(self, context: VoiceAssistantContext) -> None:
        self.logger.debug(
            "Starting listening timeout timer (%s seconds)", self._timeout_seconds
        )
        self._timeout_task = asyncio.create_task(
            self._timeout_loop(context), name="listening_timeout"
        )

    async def _stop_timeout(self) -> None:
        if self._timeout_task is None or self._timeout_task.done():
            return

        self.logger.debug("Stopping listening timeout timer")
        self._timeout_task.cancel()
        try:
            await self._timeout_task
        except asyncio.CancelledError:
            pass
        finally:
            self._timeout_task = None

    async def _timeout_loop(self, context: VoiceAssistantContext) -> None:
        try:
            await asyncio.sleep(self._timeout_seconds)
            self.logger.warning("Listening timeout occurred")
            context.event_bus.publish_sync(VoiceAssistantEvent.TIMEOUT_OCCURRED)
        except asyncio.CancelledError:
            self.logger.debug("Listening timeout cancelled")
            raise
