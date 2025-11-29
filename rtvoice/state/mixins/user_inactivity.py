import asyncio
from collections.abc import Callable

from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.state.context import VoiceAssistantContext
from rtvoice.state.events import VoiceAssistantEvent


# TODO: Name should be somethin like UserInactivityTimeoutMixin
class IdleTimeoutMixin(LoggingMixin):
    _TIMEOUT_SECONDS = 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timeout_task: asyncio.Task | None = None

    async def _start_idle_timeout(
        self,
        context: VoiceAssistantContext,
        on_timeout: Callable[[VoiceAssistantContext], None] | VoiceAssistantEvent,
        timeout_name: str = "timeout",
    ) -> None:
        self.logger.debug(
            "Starting %s timer (%.1f seconds)", timeout_name, self._TIMEOUT_SECONDS
        )
        self._timeout_task = asyncio.create_task(
            self._timeout_loop(context, on_timeout, timeout_name),
            name=f"{timeout_name}_timer",
        )

    async def _stop_idle_timeout(self) -> None:
        if self._timeout_task is None or self._timeout_task.done():
            return

        self.logger.debug("Stopping timeout timer")
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
        timeout_name: str,
    ) -> None:
        try:
            await asyncio.sleep(self._TIMEOUT_SECONDS)
            self.logger.warning("%s timeout occurred", timeout_name.capitalize())

            if isinstance(on_timeout, VoiceAssistantEvent):
                context.event_bus.publish_sync(on_timeout)
            else:
                await on_timeout(context)

        except asyncio.CancelledError:
            self.logger.debug("%s timeout cancelled", timeout_name.capitalize())
            raise
