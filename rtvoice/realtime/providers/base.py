from abc import ABC, abstractmethod


class RealtimeProvider(ABC):
    """Abstract base class for Realtime API providers.

    Implement this interface to add support for a new backend
    (e.g. a self-hosted proxy or a different cloud provider).
    """

    @abstractmethod
    def build_url(self, model: str) -> str:
        """Return the WebSocket URL for the given model identifier."""
        ...

    @abstractmethod
    def build_headers(self) -> dict[str, str]:
        """Return the HTTP headers required to authenticate the connection."""
        ...
