from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rtvoice.state.base import AssistantState, StateType, VoiceAssistantEvent

if TYPE_CHECKING:
    from rtvoice.state.base import VoiceAssistantContext


class IdleState(AssistantState):
    def __init__(self):
        super().__init__(StateType.IDLE)
        self._wake_task: asyncio.Task | None = None

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        await self._start_wake_word_detection(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._stop_wake_word_detection()

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.WAKE_WORD_DETECTED:
                await self._transition_to_listening(context)
            case _:
                self.logger.debug("Ignoring event %s in Idle state", event.value)

    async def _start_wake_word_detection(self, context: VoiceAssistantContext) -> None:
        self.logger.debug("Starting wake word detection task")
        self._wake_task = asyncio.create_task(
            self._wake_word_loop(context), name="wake_word_detection"
        )

    async def _stop_wake_word_detection(self):
        if self._wake_task is None or self._wake_task.done():
            return

        self._wake_task.cancel()
        try:
            await self._wake_task
        except asyncio.CancelledError:  # NOSONAR
            pass
        finally:
            self._wake_task = None

    async def _wake_word_loop(self, context: VoiceAssistantContext) -> None:
        try:
            self.logger.debug("Starting wake word detection...")
            await context._wake_word_listener.listen_for_wakeword()
            self.logger.debug("Wake word detection completed")

        except asyncio.CancelledError:
            self.logger.debug("Wake word detection cancelled")
            raise
