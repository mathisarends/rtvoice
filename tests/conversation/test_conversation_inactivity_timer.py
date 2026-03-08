import time

from rtvoice.conversation import ConversationInactivityTimer


class TestElapsed:
    def test_returns_zero_before_reset(self) -> None:
        timer = ConversationInactivityTimer(timeout_seconds=10.0)
        assert timer.elapsed() == 0.0

    def test_increases_after_reset(self) -> None:
        timer = ConversationInactivityTimer(timeout_seconds=10.0)
        timer.reset()
        time.sleep(0.05)
        assert timer.elapsed() >= 0.05


class TestRemaining:
    def test_returns_full_timeout_before_reset(self) -> None:
        timer = ConversationInactivityTimer(timeout_seconds=10.0)
        assert timer.remaining() == 10

    def test_decreases_over_time(self) -> None:
        timer = ConversationInactivityTimer(timeout_seconds=10.0)
        timer.reset()
        time.sleep(0.05)
        assert timer.remaining() <= 10

    def test_never_goes_below_zero(self) -> None:
        timer = ConversationInactivityTimer(timeout_seconds=0.01)
        timer.reset()
        time.sleep(0.05)
        assert timer.remaining() == 0


class TestHasTimedOut:
    def test_false_before_reset(self) -> None:
        timer = ConversationInactivityTimer(timeout_seconds=10.0)
        assert timer.has_timed_out() is False

    def test_false_immediately_after_reset(self) -> None:
        timer = ConversationInactivityTimer(timeout_seconds=10.0)
        timer.reset()
        assert timer.has_timed_out() is False

    def test_true_after_timeout_exceeded(self) -> None:
        timer = ConversationInactivityTimer(timeout_seconds=0.01)
        timer.reset()
        time.sleep(0.05)
        assert timer.has_timed_out() is True

    def test_resets_after_reset_call(self) -> None:
        timer = ConversationInactivityTimer(timeout_seconds=0.01)
        timer.reset()
        time.sleep(0.05)
        assert timer.has_timed_out() is True
        timer.reset()
        assert timer.has_timed_out() is False
