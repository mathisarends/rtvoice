from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.state.base import VoiceAssistantEvent
from rtvoice.state.context import VoiceAssistantContext

if TYPE_CHECKING:
    from rtvoice.state.base import AssistantState


class VoiceAssistantStateMachine(LoggingMixin):
    def __init__(self, context: VoiceAssistantContext):
        from rtvoice.state.idle import IdleState

        self._context = context
        self._state: AssistantState = IdleState()
        self._state.set_state_machine(self)
        self._realtime_task: asyncio.Task | None = None
        self._running = False

        self._setup_event_subscriptions()

    @property
    def state(self) -> AssistantState:
        return self._state

    @property
    def context(self) -> VoiceAssistantContext:
        return self._context

    async def run(self) -> None:
        if self._running:
            self.logger.warning("State machine already running")
            return

        self._running = True
        self._context.audio_player.play_startup_sound()
        await self._state.on_enter(self._context)

    async def stop(self) -> None:
        if not self._running:
            return

        self.logger.info("Stopping state machine...")
        self._running = False
        await self._state.on_exit(self._context)

    async def transition_to(self, new_state: AssistantState) -> None:
        self.logger.info(
            "Transitioning from %s to %s",
            self._state.state_type,
            new_state.state_type,
        )

        await self._state.on_exit(self._context)
        self._state = new_state
        self._state.set_state_machine(self)  # Set reference for new state
        await self._state.on_enter(self._context)

    def _setup_event_subscriptions(self) -> None:
        for event_type in VoiceAssistantEvent:
            self._context.event_bus.subscribe(event_type, self._handle_event)

    async def _handle_event(self, event: VoiceAssistantEvent, data: Any = None) -> None:
        await self._state.handle(event, self._context)

    async def start_realtime_session(self) -> bool:
        if self._is_realtime_session_active():
            self.logger.warning("Realtime session already active, skipping start")
            return False

        self.logger.info("Starting realtime session...")
        self._realtime_task = asyncio.create_task(
            self._context.realtime_client.setup_and_run()
        )
        self.logger.info("Realtime session started successfully")
        return True

    async def close_realtime_session(self, timeout: float = 1.0) -> bool:
        if not self._is_realtime_session_active():
            return True

        try:
            await self._context.realtime_client.close_connection()

            if self._realtime_task:
                await asyncio.wait_for(self._realtime_task, timeout=timeout)

            self._realtime_task = None
            return True

        except TimeoutError:
            self.logger.error("Task didn't complete - this should not happen!")
            if self._realtime_task:
                self._realtime_task.cancel()
            return False

    def ensure_realtime_audio_channel_paused(self) -> None:
        if not self._is_realtime_session_active():
            raise RuntimeError("Cannot pause audio - realtime session not active")

        if not self._is_realtime_audio_paused():
            self._context.realtime_client.pause_audio_streaming()
            self.logger.info("Realtime audio streaming paused")

    async def ensure_realtime_audio_channel_connected(self) -> None:
        if not self._is_realtime_session_active():
            self.logger.info("Realtime session not active, starting new session...")
            await self.start_realtime_session()

        if not self._context.audio_capture.is_active:
            self._context.audio_capture.start_stream()
            self.logger.info("Microphone stream reactivated")

        if self._is_realtime_audio_paused():
            self._context.realtime_client.resume_audio_streaming()
            self.logger.info("Realtime audio streaming resumed")

    def _is_realtime_audio_paused(self) -> bool:
        if not self._is_realtime_session_active():
            return False
        return self._context.realtime_client.is_audio_streaming_paused()

    def _is_realtime_session_active(self) -> bool:
        return self._realtime_task is not None and not self._realtime_task.done()
