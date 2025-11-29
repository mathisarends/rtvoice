from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.sound.audio import AudioStrategy


class AudioPlayer(LoggingMixin):
    def __init__(self, strategy: AudioStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: AudioStrategy) -> None:
        old_name = type(self._strategy).__name__
        new_name = type(strategy).__name__
        self.logger.info(f"Switching from {old_name} to {new_name}")
        self._strategy = strategy

    def set_volumne_level(self, volume: float) -> None:
        self._strategy.set_volume_level(volume)

    @property
    def strategy(self) -> AudioStrategy:
        return self._strategy

    def stop_sounds(self) -> None:
        self._strategy.stop_sounds()

    def clear_queue_and_stop_chunks(self) -> None:
        self._strategy.clear_queue_and_stop_chunks()

    def play_startup_sound(self) -> None:
        self._strategy.play_startup_sound()

    def play_wake_word_sound(self) -> None:
        self._strategy.play_wake_word_sound()

    def play_return_to_idle_sound(self) -> None:
        self._strategy.play_return_to_idle_sound()

    def play_error_sound(self) -> None:
        self._strategy.play_error_sound()

    def add_audio_chunk(self, base64_audio: str) -> None:
        self._strategy.add_audio_chunk(base64_audio)
