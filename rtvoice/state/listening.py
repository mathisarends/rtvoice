from rtvoice.state.base import AssistantState, StateType, VoiceAssistantEvent
from rtvoice.state.context import VoiceAssistantContext


class ListeningState(AssistantState):
    def __init__(self):
        super().__init__(StateType.LISTENING)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Entering Listening state - user is speaking")
        context._audio_player.clear_queue_and_stop_chunks()

        await context.ensure_realtime_audio_channel_connected()

        self.logger.debug("Initiating realtime session for user conversation")
        await context.start_realtime_session()

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.USER_SPEECH_ENDED:
                self.logger.info("User finished speaking")
                return await self._transition_to_responding(context)
            case _:
                self.logger.debug("Ignoring event %s in Listening state", event.value)
