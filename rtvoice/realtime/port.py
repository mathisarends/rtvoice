from abc import ABC, abstractmethod


class RealtimeProvider(ABC):
    @abstractmethod
    def build_url(self, model: str) -> str: ...

    @abstractmethod
    def build_headers(self) -> dict[str, str]: ...
