import inspect
from datetime import datetime
from typing import Annotated, Any

from rtvoice.events import EventBus
from rtvoice.events.views import (
    StopAgentCommand,
    VolumeUpdateRequestedEvent,
)
from rtvoice.mcp.server import MCPServer
from rtvoice.realtime.schemas import FunctionTool
from rtvoice.shared.logging import LoggingMixin
from rtvoice.tools.registry import ToolRegistry
from rtvoice.tools.registry.views import Tool
from rtvoice.tools.views import SpecialToolParameters
from rtvoice.views import ActionResult


class Tools(LoggingMixin):
    def __init__(self):
        self._registry = ToolRegistry()
        self._context: SpecialToolParameters = SpecialToolParameters()
        self._register_default_tools()

    def set_context(self, context: SpecialToolParameters) -> None:
        self._context = context

    def action(self, description: str, **kwargs):
        return self._registry.action(description, **kwargs)

    def register_mcp(self, tool: FunctionTool, server: MCPServer) -> None:
        self._registry.register_mcp(tool, server)

    def get_tool_schema(self) -> list[FunctionTool]:
        return self._registry.get_tool_schema()

    def get_json_tool_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": tool.model_dump(exclude={"type"}, exclude_none=True),
            }
            for tool in self._registry.get_tool_schema()
        ]

    def get(self, name: str) -> Tool | None:
        return self._registry.get(name)

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        tool = self._registry.get(name)
        if not tool:
            raise KeyError(f"Tool '{name}' not found in registry")

        prepared = self._prepare_arguments(tool, arguments, self._context)
        return await tool.execute(prepared)

    def _prepare_arguments(
        self,
        tool: Tool,
        llm_arguments: dict[str, Any],
        context: SpecialToolParameters,
    ) -> dict[str, Any]:
        signature = inspect.signature(tool.function)
        arguments = llm_arguments.copy()
        injectable = self._injectable_from_context(context)

        for param_name, param in signature.parameters.items():
            if param_name in arguments or param_name in ("self", "cls"):
                continue

            if param_name in injectable and injectable[param_name] is not None:
                arguments[param_name] = injectable[param_name]
            elif param.default is inspect.Parameter.empty:
                raise ValueError(
                    f"Missing required parameter '{param_name}' for tool '{tool.name}'"
                )

        return arguments

    def _injectable_from_context(
        self, context: SpecialToolParameters
    ) -> dict[str, Any]:
        return {field: getattr(context, field) for field in context.model_fields}

    def _register_default_tools(self) -> None:
        @self.action("Get the current local time")
        def get_current_time() -> str:
            return datetime.now().strftime("%H:%M:%S")

        @self.action("Adjust volume level.")
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

        @self.action("Stop the current realtime session.")
        async def stop_session(event_bus: EventBus) -> ActionResult:
            self.logger.info("Stop command received - dispatching stop event")

            stop_event = StopAgentCommand()
            await event_bus.dispatch(stop_event)

            return ActionResult(success=True, message="Stopping agent session")
