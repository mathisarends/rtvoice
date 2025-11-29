from rtvoice.state.base import AssistantState, StateType, VoiceAssistantEvent
from rtvoice.state.context import VoiceAssistantContext
from rtvoice.state.timeout_service import TimeoutService


class TimeoutState(AssistantState):
    """State after wake word - waiting for user to start speaking with timeout"""

    def __init__(self):
        super().__init__(StateType.TIMEOUT)
        self.timeout_service = None
        self._event_handlers = {
            VoiceAssistantEvent.USER_STARTED_SPEAKING: self._handle_user_started_speaking,
            VoiceAssistantEvent.TIMEOUT_OCCURRED: self._handle_timeout,
        }

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        if self.timeout_service is None:
            self.timeout_service = TimeoutService(
                timeout_seconds=10.0, event_bus=context._event_bus
            )

        self.logger.info(
            "Entering TimeoutState - user has %s seconds to start speaking",
            self.timeout_service.timeout_seconds,
        )
        await context.ensure_realtime_audio_channel_connected()

        # Start both timeout service and audio detection
        await self._start_timeout_service(context)
        await self._start_audio_detection(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._stop_timeout_service(context)
        await self._stop_audio_detection(context)

        from rtvoice.state.base import StateType

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
            self.timeout_service.timeout_seconds,
        )
        await self.transition_to_idle(context)

    async def _start_timeout_service(self, context: VoiceAssistantContext) -> None:
        self.logger.debug("Starting timeout service")
        if self.timeout_service:
            await self.timeout_service.start_timeout()

    async def _stop_timeout_service(self, context: VoiceAssistantContext) -> None:
        self.logger.debug("Stopping timeout service")
        if self.timeout_service:
            await self.timeout_service.stop_timeout()
            self.timeout_service = None

    async def _start_audio_detection(self, context: VoiceAssistantContext) -> None:
        self.logger.debug("Starting audio detection for speech detection")
        await context._speech_detector.start_monitoring()

    async def _stop_audio_detection(self, context: VoiceAssistantContext) -> None:
        self.logger.debug("Stopping audio detection")
        await context._speech_detector.stop_monitoring()
