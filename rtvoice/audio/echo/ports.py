from abc import ABC, abstractmethod
from collections.abc import Callable

Clock = Callable[[], float]


class EchoCanceller(ABC):
    """Removes the far-end (loudspeaker) signal from near-end (microphone) audio.

    Implementations see nothing but PCM, which keeps them independent of how the
    audio is captured or played back.
    """

    @abstractmethod
    def process(self, near_end: bytes, far_end: bytes) -> bytes:
        """`near_end` and `far_end` are equally long mono 16-bit PCM frames, where
        `far_end` is what the speaker rendered while `near_end` was captured.

        Implementations that work on fixed-size blocks may return fewer samples than
        they were handed; callers treat the result as a continuous stream.
        """

    def reset(self) -> None:  # noqa: B027 - optional hook, stateless filters need none
        """Drop adaptation state, e.g. when capture restarts on another device."""
