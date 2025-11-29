import asyncio

from rtvoice.state.base import AssistantState, VoiceAssistantEvent
from rtvoice.state.context import VoiceAssistantContext
from rtvoice.state.models import StateType


class TimeoutState(AssistantState):
    def __init__(self, timeout_seconds: float = 10.0):
        self._timeout_seconds = timeout_seconds
        self._timeout_task: asyncio.Task | None = None
        self._event_handlers = {
            VoiceAssistantEvent.USER_STARTED_SPEAKING: self._handle_user_started_speaking,
            VoiceAssistantEvent.TIMEOUT_OCCURRED: self._handle_timeout,
        }

    @property
    def state_type(self) -> StateType:
        return StateType.TIMEOUT

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info(
            "Entering TimeoutState - user has %s seconds to start speaking",
            self._timeout_seconds,
        )
        await context.ensure_realtime_audio_channel_connected()

        await self._start_timeout(context)
        await self._start_audio_detection(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._stop_timeout()
        await self._stop_audio_detection(context)

        if context._state.state_type == StateType.IDLE:
            context._event_bus.publish_sync(VoiceAssistantEvent.IDLE_TRANSITION)
            self.logger.info("Closing realtime connection due to timeout")
            await context._realtime_client.close_connection()

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        handler = self._event_handlers.get(event)
        if handler:
            await handler(context)

    async def _handle_user_started_speaking(
        self, context: VoiceAssistantContext
    ) -> None:
        self.logger.info("User started speaking - transitioning to listening")
        await self._transition_to_listening(context)

    async def _handle_timeout(self, context: VoiceAssistantContext) -> None:
        self.logger.info(
            "Timeout occurred - user did not start speaking within %s seconds",
            self._timeout_seconds,
        )
        await self.transition_to_idle(context)

    async def _start_timeout(self, context: VoiceAssistantContext) -> None:
        self.logger.debug("Starting timeout timer")
        self._timeout_task = asyncio.create_task(
            self._timeout_loop(context), name="timeout_timer"
        )

    async def _stop_timeout(self) -> None:
        if self._timeout_task is None or self._timeout_task.done():
            return

        self.logger.debug("Stopping timeout timer")
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
            context._event_bus.publish_sync(VoiceAssistantEvent.TIMEOUT_OCCURRED)
        except asyncio.CancelledError:
            self.logger.debug("Timeout cancelled")
            raise

    async def _start_audio_detection(self, context: VoiceAssistantContext) -> None:
        self.logger.debug("Starting audio detection for speech detection")
        await context._speech_detector.start_monitoring()

    async def _stop_audio_detection(self, context: VoiceAssistantContext) -> None:
        self.logger.debug("Stopping audio detection")
        await context._speech_detector.stop_monitoring()
