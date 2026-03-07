from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class AudioInputDevice(ABC):
    """Abstract base class for audio input devices.

    Implement this interface to provide a custom microphone or audio source.
    The default implementation is [MicrophoneInput][rtvoice.audio.MicrophoneInput].
    """

    @abstractmethod
    async def start(self) -> None:
        """Open the device and begin audio capture."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop audio capture and release all device resources."""

    @abstractmethod
    def stream_chunks(self) -> AsyncIterator[bytes]:
        """Yield raw 16-bit PCM audio chunks as they become available."""

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Whether the device is currently capturing audio."""


class AudioOutputDevice(ABC):
    """Abstract base class for audio output devices.

    Implement this interface to provide a custom speaker or audio sink.
    The default implementation is [SpeakerOutput][rtvoice.audio.SpeakerOutput].
    """

    @abstractmethod
    async def start(self) -> None:
        """Open the device and prepare it for playback."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop playback and release all device resources."""

    @abstractmethod
    async def play_chunk(self, chunk: bytes) -> None:
        """Enqueue a raw 16-bit PCM audio chunk for playback."""

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """Whether audio is currently being played or queued."""

    @abstractmethod
    async def clear_buffer(self) -> None:
        """Discard all queued audio to stop playback immediately."""
