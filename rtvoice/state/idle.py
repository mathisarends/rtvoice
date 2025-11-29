from __future__ import annotations

import asyncio

from rtvoice.state.base import AssistantState
from rtvoice.state.context import VoiceAssistantContext
from rtvoice.state.events import VoiceAssistantEvent
from rtvoice.state.models import StateType


class IdleState(AssistantState):
    def __init__(self):
        super().__init__()
        self._wake_task: asyncio.Task | None = None
        self._event_handlers = {
            VoiceAssistantEvent.WAKE_WORD_DETECTED: self._handle_wake_word,
        }

    @property
    def state_type(self) -> StateType:
        return StateType.IDLE

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        await self._start_wake_word_detection(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._stop_wake_word_detection()

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        handler = self._event_handlers.get(event)
        if handler:
            await handler(context)

    async def _handle_wake_word(self, context: VoiceAssistantContext) -> None:
        await self._transition_to_listening()

    async def _start_wake_word_detection(self, context: VoiceAssistantContext) -> None:
        self.logger.debug("Starting wake word detection task")
        self._wake_task = asyncio.create_task(
            self._wake_word_loop(context), name="wake_word_detection"
        )

    async def _stop_wake_word_detection(self) -> None:
        if self._wake_task is None or self._wake_task.done():
            return

        self._wake_task.cancel()
        try:
            await self._wake_task
        except asyncio.CancelledError:
            pass
        finally:
            self._wake_task = None

    async def _wake_word_loop(self, context: VoiceAssistantContext) -> None:
        try:
            self.logger.debug("Starting wake word detection...")
            await context.wake_word_listener.listen_for_wakeword()
            self.logger.debug("Wake word detection completed")

        except asyncio.CancelledError:
            self.logger.debug("Wake word detection cancelled")
            raise
