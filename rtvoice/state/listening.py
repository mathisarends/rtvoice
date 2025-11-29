from rtvoice.state.base import AssistantState, VoiceAssistantEvent
from rtvoice.state.context import VoiceAssistantContext
from rtvoice.state.models import StateType


class ListeningState(AssistantState):
    def __init__(self):
        self._event_handlers = {
            VoiceAssistantEvent.USER_SPEECH_ENDED: self._handle_speech_ended,
        }

    @property
    def state_type(self) -> StateType:
        return StateType.LISTENING

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Entering Listening state - user is speaking")
        context._audio_player.clear_queue_and_stop_chunks()

        await context.ensure_realtime_audio_channel_connected()

        self.logger.debug("Initiating realtime session for user conversation")
        await context.start_realtime_session()

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        handler = self._event_handlers.get(event)
        if handler:
            await handler(context)

    async def _handle_speech_ended(self, context: VoiceAssistantContext) -> None:
        self.logger.info("User finished speaking")
        await self._transition_to_responding(context)
