import abc
from collections.abc import AsyncIterator


class AudioInputDevice(abc.ABC):
    @abc.abstractmethod
    async def start(self) -> None:
        """Open the device and begin capture."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop capture and release resources."""

    @abc.abstractmethod
    def stream_chunks(self) -> AsyncIterator[bytes]:
        """Yield raw 16-bit PCM chunks."""

    @property
    @abc.abstractmethod
    def is_active(self) -> bool:
        """Whether the device is currently capturing."""


class AudioOutputDevice(abc.ABC):
    @abc.abstractmethod
    async def start(self) -> None:
        """Open the device and prepare for playback."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop playback and release resources."""

    @abc.abstractmethod
    async def play_chunk(self, chunk: bytes) -> None:
        """Enqueue a raw 16-bit PCM chunk."""

    @property
    @abc.abstractmethod
    def is_playing(self) -> bool:
        """Whether audio is currently playing or queued."""

    @abc.abstractmethod
    async def clear_buffer(self) -> None:
        """Discard all queued audio immediately."""
