from rtvoice.mic.inactivity_timer import UserSpeechInactivityTimer
from rtvoice.state.base import AssistantState, VoiceAssistantEvent
from rtvoice.state.context import VoiceAssistantContext
from rtvoice.state.models import StateType


class TimeoutState(AssistantState):
    def __init__(
        self, user_speech_inactivity_timer: UserSpeechInactivityTimer | None = None
    ):
        super().__init__()
        self._user_speech_inactivity_timer = (
            user_speech_inactivity_timer or UserSpeechInactivityTimer()
        )
        self._event_handlers = {
            VoiceAssistantEvent.USER_STARTED_SPEAKING: self._handle_user_started_speaking,
        }

    @property
    def state_type(self) -> StateType:
        return StateType.TIMEOUT

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info(
            "Entering TimeoutState - user has 10 seconds to start speaking"
        )
        await self._state_machine.ensure_realtime_audio_channel_connected()

        await self._user_speech_inactivity_timer.start(
            context,
            on_timeout=self._handle_timeout,
        )

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._user_speech_inactivity_timer.stop()

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

    def _handle_timeout(self) -> None:
        """Called by timer when timeout occurs."""
        self.logger.info(
            "Timeout occurred - user did not start speaking within 10 seconds"
        )
        # Schedule transition in event loop
        import asyncio

        asyncio.create_task(self._transition_to_idle())
