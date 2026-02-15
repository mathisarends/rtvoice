from datetime import datetime
from typing import Annotated

from rtvoice.events import EventBus
from rtvoice.events.views import (
    StopAgentCommand,
    VolumeUpdateRequestedEvent,
)
from rtvoice.realtime.schemas import FunctionTool
from rtvoice.shared.logging import LoggingMixin
from rtvoice.tools.registry import ToolRegistry
from rtvoice.views import ActionResult


class Tools(LoggingMixin):
    def __init__(self):
        self.registry = ToolRegistry()

        self._register_default_tools()

    def action(self, description: str, **kwargs):
        return self.registry.action(description, **kwargs)

    def get_schema(self) -> list[FunctionTool]:
        return self.registry.get_schema()

    def _register_default_tools(self) -> None:
        @self.registry.action("Get the current local time")
        def get_current_time() -> str:
            return datetime.now().strftime("%H:%M:%S")

        @self.registry.action("Adjust volume level.")
        async def adjust_volume(
            level: Annotated[float, "Volume level from 0.0 (0%) to 1.0 (100%)"],
            event_bus: EventBus,
        ) -> ActionResult:
            clamped_level = max(0.0, min(1.0, level))

            if level != clamped_level:
                self.logger.warning(
                    "Volume level %.2f out of range, clamped to %.2f",
                    level,
                    clamped_level,
                )

            event = VolumeUpdateRequestedEvent(volume=clamped_level)
            await event_bus.dispatch(event)

            percentage = int(clamped_level * 100)
            return ActionResult(
                success=True, message=f"Volume adjusted to {percentage}%"
            )

        @self.registry.action("Stop the current realtime session.")
        async def stop_realtime_session(event_bus: EventBus) -> ActionResult:
            self.logger.info("Stop command received - dispatching stop event")

            stop_event = StopAgentCommand()
            await event_bus.dispatch(stop_event)

            return ActionResult(success=True, message="Stopping agent session")
