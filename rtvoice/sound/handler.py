from rtvoice.events import EventBus
from rtvoice.events.schemas import ResponseOutputAudioDeltaEvent
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.sound.player import AudioPlayer
from rtvoice.state.base import VoiceAssistantEvent


class SoundEventHandler(LoggingMixin):
    def __init__(self, audio_player: AudioPlayer, event_bus: EventBus):
        self._audio_manager = audio_player
        self._event_bus = event_bus

        self._subscribe_to_events()
        self.logger.info("SoundEventHandler initialized and subscribed to events")

    def _subscribe_to_events(self):
        self._event_bus.subscribe(
            VoiceAssistantEvent.AUDIO_CHUNK_RECEIVED, self._handle_audio_chunk_event
        )
        self._event_bus.subscribe(
            VoiceAssistantEvent.WAKE_WORD_DETECTED, self._handle_wake_word_event
        )
        self._event_bus.subscribe(
            VoiceAssistantEvent.IDLE_TRANSITION, self._handle_idle_transition_event
        )
        self._event_bus.subscribe(
            VoiceAssistantEvent.ERROR_OCCURRED, self._handle_error_event
        )
        self._event_bus.subscribe(
            VoiceAssistantEvent.USER_STARTED_SPEAKING,
            self._handle_user_started_speaking,
        )

    def _handle_audio_chunk_event(
        self,
        response_output_audio_delta: ResponseOutputAudioDeltaEvent,
    ) -> None:
        self.logger.debug("Received audio chunk via EventBus")
        self._audio_manager.add_audio_chunk(response_output_audio_delta.delta)

    def _handle_wake_word_event(self) -> None:
        self.logger.debug("Playing wake word sound via EventBus")
        self._audio_manager.play_wake_word_sound()

    def _handle_idle_transition_event(self) -> None:
        self.logger.debug("Playing return to idle sound via EventBus")
        self._audio_manager.play_return_to_idle_sound()

    def _handle_error_event(self) -> None:
        self.logger.debug("Playing error sound via EventBus")
        self._audio_manager.play_error_sound()

    def _handle_user_started_speaking(self) -> None:
        if self._audio_manager.strategy.is_currently_playing_chunks():
            self._event_bus.publish_sync(
                VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED
            )

        self.logger.debug("User started speaking, clearing audio queue")
        self._audio_manager.strategy.clear_queue_and_stop_chunks()
