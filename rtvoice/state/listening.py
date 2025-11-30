from rtvoice.mic.inactivity_timer import UserSpeechInactivityTimer
from rtvoice.state.base import AssistantState
from rtvoice.state.context import VoiceAssistantContext
from rtvoice.state.events import VoiceAssistantEvent
from rtvoice.state.models import StateType


class ListeningState(AssistantState):
    def __init__(
        self, user_speech_inactivity_timer: UserSpeechInactivityTimer | None = None
    ):
        super().__init__()
        self._user_speech_inactivity_timer = (
            user_speech_inactivity_timer or UserSpeechInactivityTimer()
        )
        self._event_handlers = {
            VoiceAssistantEvent.USER_SPEECH_ENDED: self._handle_speech_ended,
            VoiceAssistantEvent.IDLE_TRANSITION: self._handle_idle_transition,
        }

    @property
    def state_type(self) -> StateType:
        return StateType.LISTENING

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Entering Listening state - user is speaking")
        context.audio_player.clear_queue_and_stop_chunks()

        await self._state_machine.ensure_realtime_audio_channel_connected()

        await self._user_speech_inactivity_timer.start(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._user_speech_inactivity_timer.stop()

        if self._state_machine.is_transitioning_to_idle_state():
            self.logger.info("Closing realtime connection due to idle transition")
            await self._state_machine.close_realtime_session()

    async def handle(self, event: VoiceAssistantEvent) -> None:
        handler = self._event_handlers.get(event)
        if handler:
            await handler()

    async def _handle_speech_ended(self) -> None:
        self.logger.info("User finished speaking")
        await self._transition_to_responding()

    async def _handle_idle_transition(self) -> None:
        await self._transition_to_idle()
