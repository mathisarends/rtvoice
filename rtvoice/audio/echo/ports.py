from abc import ABC, abstractmethod
from collections.abc import Callable

type Clock = Callable[[], float]


class EchoCanceller(ABC):
    @abstractmethod
    def process(self, near_end: bytes, far_end: bytes) -> bytes: ...

    def reset(self) -> None:  # noqa: B027 - optional hook, stateless filters need none
        pass
