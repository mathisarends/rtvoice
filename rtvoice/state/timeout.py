from rtvoice.state.base import AssistantState, VoiceAssistantEvent
from rtvoice.state.context import VoiceAssistantContext
from rtvoice.state.mixins import IdleTimeoutMixin
from rtvoice.state.models import StateType


class TimeoutState(IdleTimeoutMixin, AssistantState):
    def __init__(self):
        super().__init__()
        self._event_handlers = {
            VoiceAssistantEvent.USER_STARTED_SPEAKING: self._handle_user_started_speaking,
            VoiceAssistantEvent.TIMEOUT_OCCURRED: self._handle_timeout,
        }

    @property
    def state_type(self) -> StateType:
        return StateType.TIMEOUT

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info(
            "Entering TimeoutState - user has 10 seconds to start speaking"
        )
        await self._state_machine.ensure_realtime_audio_channel_connected()

        await self._start_idle_timeout(
            context,
            on_timeout=VoiceAssistantEvent.TIMEOUT_OCCURRED,
            timeout_name="user_speech_timeout",
        )
        await self._start_audio_detection(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._stop_idle_timeout()
        await self._stop_audio_detection(context)

        if self._state_machine.state.state_type == StateType.IDLE:
            context.event_bus.publish_sync(VoiceAssistantEvent.IDLE_TRANSITION)
            self.logger.info("Closing realtime connection due to timeout")
            await self._state_machine.close_realtime_session()

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
        await self._transition_to_listening()

    async def _handle_timeout(self, context: VoiceAssistantContext) -> None:
        self.logger.info(
            "Timeout occurred - user did not start speaking within 10 seconds"
        )
        await self._transition_to_idle()

    async def _start_audio_detection(self, context: VoiceAssistantContext) -> None:
        self.logger.debug("Starting audio detection for speech detection")
        await context.speech_detector.start_monitoring()

    async def _stop_audio_detection(self, context: VoiceAssistantContext) -> None:
        self.logger.debug("Stopping audio detection")
        await context.speech_detector.stop_monitoring()
