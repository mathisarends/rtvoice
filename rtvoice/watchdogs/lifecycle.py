import asyncio
import logging
from contextlib import suppress

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentSessionConnectedEvent,
    AgentStoppedEvent,
    ConfigureSessionCommand,
    StartAgentCommand,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.shared.decorators import timed

logger = logging.getLogger(__name__)


class LifecycleWatchdog:
    def __init__(self, event_bus: EventBus, websocket: RealtimeWebSocket):
        self._event_bus = event_bus
        self._websocket = websocket
        self._forward_task: asyncio.Task | None = None

        event_bus.subscribe(StartAgentCommand, self._on_start_agent_command)
        event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)

    @timed()
    async def _on_start_agent_command(self, command: StartAgentCommand) -> None:
        logger.info("Starting agent session")

        if not self._websocket.is_connected:
            await self._websocket.connect()

        self._forward_task = asyncio.create_task(self._forward_events())

        await self._event_bus.dispatch(
            ConfigureSessionCommand(
                model=command.model,
                instructions=command.instructions,
                voice=command.voice,
                speech_speed=command.speech_speed,
                transcription_model=command.transcription_model,
                output_modalities=command.output_modalities,
                noise_reduction=command.noise_reduction,
                turn_detection=command.turn_detection,
                tools=command.tools,
            )
        )
        await self._event_bus.dispatch(AgentSessionConnectedEvent())

        logger.info("Agent session ready")

    async def _forward_events(self) -> None:
        async for event in self._websocket.events():
            await self._event_bus.dispatch(event)

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        if self._forward_task and not self._forward_task.done():
            self._forward_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._forward_task

        if not self._websocket.is_connected:
            return

        await self._websocket.close()
        logger.info("Agent session stopped")
