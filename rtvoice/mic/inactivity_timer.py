import asyncio
from collections.abc import Callable

from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.state.context import VoiceAssistantContext
from rtvoice.state.events import VoiceAssistantEvent


class UserSpeechInactivityTimer(LoggingMixin):
    def __init__(self, timeout_seconds: float = 10.0):
        self._timeout_seconds = timeout_seconds
        self._timeout_task: asyncio.Task | None = None

    async def start(
        self,
        context: VoiceAssistantContext,
        on_timeout: Callable[[VoiceAssistantContext], None] | VoiceAssistantEvent,
    ) -> None:
        self.logger.debug("Starting timer (%.1f seconds)", self._timeout_seconds)
        self._timeout_task = asyncio.create_task(
            self._timeout_loop(context, on_timeout),
            name="user_speech_inactivity_timer",
        )

    async def stop(self) -> None:
        if self._timeout_task is None or self._timeout_task.done():
            return

        self.logger.debug("Stopping timer")
        self._timeout_task.cancel()
        try:
            await self._timeout_task
        except asyncio.CancelledError:
            pass
        finally:
            self._timeout_task = None

    async def _timeout_loop(
        self,
        context: VoiceAssistantContext,
        on_timeout: Callable[[VoiceAssistantContext], None] | VoiceAssistantEvent,
    ) -> None:
        try:
            await asyncio.sleep(self._timeout_seconds)
            self.logger.warning("Timeout occurred")

            if isinstance(on_timeout, VoiceAssistantEvent):
                context.event_bus.publish_sync(on_timeout)
            else:
                await on_timeout(context)

        except asyncio.CancelledError:
            self.logger.debug("Timeout cancelled")
            raise
