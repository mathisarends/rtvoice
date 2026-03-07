from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Annotated, Any

from rtvoice.events import EventBus

if TYPE_CHECKING:
    from rtvoice.supervisor import SupervisorAgent


from rtvoice.conversation import ConversationHistory
from rtvoice.mcp.server import MCPServer
from rtvoice.realtime.schemas import FunctionTool
from rtvoice.tools.registry import ToolRegistry
from rtvoice.tools.registry.views import Tool
from rtvoice.tools.views import SpecialToolParameters

logger = logging.getLogger(__name__)


class Tools:
    def __init__(self):
        self._registry = ToolRegistry()
        self._context: SpecialToolParameters = SpecialToolParameters()

    def set_context(self, context: SpecialToolParameters) -> None:
        self._context = context

    def action(
        self,
        description: str,
        name: str | None = None,
        result_instruction: str | None = None,
        is_long_running: bool = False,
        holding_instruction: str | None = None,
    ):
        return self._registry.action(
            description,
            name=name,
            result_instruction=result_instruction,
            is_long_running=is_long_running,
            holding_instruction=holding_instruction,
        )

    def register_mcp(self, tool: FunctionTool, server: MCPServer) -> None:
        self._registry.register_mcp(tool, server)

    def register_supervisor_agent(self, agent: SupervisorAgent) -> None:
        async def _handoff(
            task: Annotated[
                str,
                """
                The task or question to delegate to this agent.
                Be specific and add enough context for the agent to complete the task without further clarification.
                """,
            ],
            event_bus: EventBus,
            conversation_history: ConversationHistory,
        ) -> str:
            agent.set_event_bus(event_bus)
            context = conversation_history.format() if conversation_history else None
            result = await agent.run(task, context=context)
            return result.message or ""

        description = agent.description
        if agent.handoff_instructions:
            description = f"{agent.description}\n\nHandoff instructions: {agent.handoff_instructions}"

        safe_name = agent.name.replace(" ", "_")
        self._registry.action(
            description,
            name=safe_name,
            result_instruction=agent.result_instructions,
            is_long_running=True,
            holding_instruction=agent.holding_instruction,
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

    def clone(self) -> Tools:
        new = Tools()
        new._registry.tools = self._registry.tools.copy()
        return new
