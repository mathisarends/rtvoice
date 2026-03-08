import logging
import time

logger = logging.getLogger(__name__)


class ConversationInactivityTimer:
    def __init__(self, timeout_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        self._last_reset_at: float | None = None

    def reset(self) -> None:
        self._last_reset_at = time.monotonic()

    def elapsed(self) -> float:
        if self._last_reset_at is None:
            return 0.0
        return time.monotonic() - self._last_reset_at

    def remaining(self) -> int:
        return max(0, int(self._timeout_seconds - self.elapsed()))

    def has_timed_out(self) -> bool:
        return self.elapsed() > self._timeout_seconds
