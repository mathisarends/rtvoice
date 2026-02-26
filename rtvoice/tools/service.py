from __future__ import annotations

import asyncio
import inspect
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any

from rtvoice.events import EventBus

if TYPE_CHECKING:
    from rtvoice.subagents import SubAgent

import logging

from rtvoice.events.views import (
    StopAgentCommand,
    SubAgentCalledEvent,
)
from rtvoice.mcp.server import MCPServer
from rtvoice.realtime.schemas import FunctionTool
from rtvoice.tools.registry import ToolRegistry
from rtvoice.tools.registry.views import Tool
from rtvoice.tools.views import SpecialToolParameters
from rtvoice.views import ActionResult

logger = logging.getLogger(__name__)


class Tools:
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

    # TODO: Das hier mit dem fire und forget sollte hier so dann nicht funktioneron
    def register_subagent(self, agent: SubAgent) -> None:
        async def _handoff(
            task: Annotated[
                str,
                """
                The task or question to delegate to this agent.
                Be specific and add enough context for the agent to complete the task without further clarification.
                """,
            ],
            event_bus: EventBus,
        ) -> str:
            await event_bus.dispatch(
                SubAgentCalledEvent(agent_name=agent.name, task=task)
            )

            if agent.fire_and_forget:
                asyncio.create_task(agent.run(task))
                return (
                    agent.result_instructions
                    or "The task has been delegated to the agent and will be completed shortly."
                )
            else:
                result = await agent.run(task)
                return result.message or ""

        description = agent.description
        if agent.handoff_instructions:
            description = f"{agent.description}\n\nHandoff instructions: {agent.handoff_instructions}"

        safe_name = agent.name.replace(" ", "_")
        result_instructions = agent.result_instructions
        self._registry.action(
            description, name=safe_name, result_instruction=result_instructions
        )(_handoff)

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
        return {
            field: getattr(context, field)
            for field in SpecialToolParameters.model_fields
        }

    def _register_default_tools(self) -> None:
        @self.action("Get the current local time")
        def get_current_time() -> str:
            return datetime.now().strftime("%H:%M:%S")

        @self.action("Stop the current realtime session.")
        async def stop_session(event_bus: EventBus) -> ActionResult:
            logger.info("Stop command received - dispatching stop event")

            stop_event = StopAgentCommand()
            await event_bus.dispatch(stop_event)

            return ActionResult(success=True, message="Stopping agent session")
