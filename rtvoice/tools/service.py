from datetime import datetime
from typing import Annotated

from rtvoice.events import EventBus
from rtvoice.shared.logging import LoggingMixin
from rtvoice.tools.models import FunctionTool, MCPTool
from rtvoice.tools.registry import ToolRegistry


class Tools(LoggingMixin):
    def __init__(self, mcp_tools: list[MCPTool] | None = None):
        self.mcp_tools = mcp_tools
        self.registry = ToolRegistry(mcp_tools=mcp_tools)
        self._register_default_tools()

    def action(self, description: str, **kwargs):
        return self.registry.action(description, **kwargs)

    def add_mcp_tools(self, tools: list[FunctionTool]) -> None:
        self.registry.add_mcp_tools(tools)

    def get_schema(self) -> list[FunctionTool | MCPTool]:
        return self.registry.get_schema()

    # TODO: Hier alles nur über den event bus als special param (andererseits kann ich den auch einfach hier injecten (- special param logic aber schon nice))
    def _register_default_tools(self) -> None:
        @self.registry.action("Get the current local time")
        def get_current_time() -> str:
            return datetime.now().strftime("%H:%M:%S")

        @self.registry.action("Adjust volume level.")
        def adjust_volume(
            level: Annotated[float, "Volume level from 0.0 (0%) to 1.0 (100%)"],
        ) -> None: ...

        @self.registry.action(
            description=(
                "Change the assistant's talking speed by a relative amount. "
                "Acknowledge the change before calling the tool. The tool "
                "internally retrieves the current speed and adjusts it relative "
                "to the current rate."
            ),
            response_instruction=(
                "State that the response speed has been adjusted and name the "
                "new speed in percent (e.g. 1.5 = 150%)"
            ),
        )
        async def change_assistant_response_speed(
            instructions: Annotated[
                str, "Natural language command: 'faster' or 'slower'"
            ],
        ) -> str: ...

        @self.registry.action(
            "Stop the assistant run. Call this when the user says 'stop', "
            "'cancel', or 'abort'."
        )
        async def stop_assistant_run(event_bus: EventBus) -> None: ...
