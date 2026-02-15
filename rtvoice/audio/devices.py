from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class AudioInputDevice(ABC):
    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    def stream_chunks(self) -> AsyncIterator[bytes]:
        pass

    @property
    @abstractmethod
    def is_active(self) -> bool:
        pass


class AudioOutputDevice(ABC):
    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    async def play_chunk(self, chunk: bytes) -> None:
        pass

    @abstractmethod
    async def set_volume(self, volume: float) -> None:
        """Set the output volume level (0.0 to 1.0)."""
        pass

    @abstractmethod
    async def clear_buffer(self) -> None:
        """Clear any queued audio data to stop playback immediately."""
        pass
