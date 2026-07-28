from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class AudioInput(ABC):
    @abstractmethod
    async def start(self) -> None:
        """Open the device and begin capture."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop capture and release resources."""

    @abstractmethod
    def stream_chunks(self) -> AsyncIterator[bytes]:
        """Yield raw 16-bit PCM chunks."""

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Whether the device is currently capturing."""


class AudioOutput(ABC):
    @abstractmethod
    async def start(self) -> None:
        """Open the device and prepare for playback."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop playback and release resources."""

    @abstractmethod
    async def play_chunk(self, chunk: bytes) -> None:
        """Enqueue a raw 16-bit PCM chunk."""

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """Whether audio is currently playing or queued."""

    @abstractmethod
    async def clear_buffer(self) -> None:
        """Discard all queued audio immediately."""
