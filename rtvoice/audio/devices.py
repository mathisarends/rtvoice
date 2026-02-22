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

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        pass

    @abstractmethod
    async def clear_buffer(self) -> None:
        """Clear any queued audio data to stop playback immediately."""
        pass
