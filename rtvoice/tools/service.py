from __future__ import annotations

import inspect
import logging
from typing import Any, Self

from rtvoice.mcp.server import MCPServer
from rtvoice.realtime.schemas import FunctionTool
from rtvoice.tools.registry import ToolRegistry
from rtvoice.tools.registry.views import Tool
from rtvoice.tools.views import SpecialToolParameters

logger = logging.getLogger(__name__)

__all__ = [
    "AgentTools",
    "RealtimeTools",
    "Tools",
]


class Tools:
    """Base class for managing and executing tools exposed to the model.

    `Tools` acts as a registry for callable actions that the model can invoke
    during a session. Use the [`action`][rtvoice.tools.Tools.action] decorator
    to register your own tools, then pass the instance to `RealtimeAgent`.

    Example:
        ```python
        tools = Tools()


        @tools.action("Get the current time in a given timezone")
        async def get_time(timezone: str) -> str: ...


        agent = RealtimeAgent(tools=tools)
        ```
    """

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
        """Register a function as a tool the model can call.

        Decorate any async function with this to make it available to the model.
        Parameters annotated with [`EventBus`][rtvoice.events.EventBus],
        [`ConversationHistory`][rtvoice.conversation.ConversationHistory], or the
        shared `context` object are injected automatically — do not include them
        in the model-facing schema.

        Args:
            description: Natural-language description shown to the model.
                Write this as an instruction, e.g. *"Get the current weather
                for a given city"*.
            name: Override the tool name exposed to the model. Defaults to the
                function name.
            result_instruction: Optional instruction appended to the tool result
                telling the model how to interpret or present the output.
            is_long_running: Set to `True` for tools that take more than a second
                or two. Enables a holding message so the assistant can inform the
                user that it is working.
            holding_instruction: Message spoken by the assistant while a
                long-running tool is executing. Requires `is_long_running=True`.

        Returns:
            A decorator that registers the decorated function and returns it unchanged.

        Example:
            ```python
            tools = Tools()


            @tools.action(
                "Get the current weather for a given city",
                result_instruction="Summarise the weather in one sentence.",
            )
            async def get_weather(city: str) -> str: ...


            @tools.action(
                "Run a slow background job",
                is_long_running=True,
                holding_instruction="Give me a moment, I'm running the job now.",
            )
            async def slow_job(task: str) -> str: ...
            ```
        """
        return self._registry.action(
            description,
            name=name,
            result_instruction=result_instruction,
            is_long_running=is_long_running,
            holding_instruction=holding_instruction,
        )

    def register_mcp(self, tool: FunctionTool, server: MCPServer) -> None:
        self._registry.register_mcp(tool, server)

    def get(self, name: str) -> Tool | None:
        """Look up a registered tool by name.

        Args:
            name: The tool name as registered (or overridden via `action(name=...)`).

        Returns:
            The [`Tool`][rtvoice.tools.registry.views.Tool] instance, or `None`
            if no tool with that name exists.
        """
        return self._registry.get(name)

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a registered tool by name with the given arguments.

        Special parameters (`event_bus`, `conversation_history`, `context`) are
        injected automatically from the shared context — do not include them in
        `arguments`.

        Args:
            name: Name of the tool to execute.
            arguments: Arguments provided by the model, as a plain dict.

        Returns:
            The return value of the tool function.

        Raises:
            KeyError: If no tool with the given name is registered.
            ValueError: If a required parameter cannot be resolved from the
                model arguments or the injected context.
        """
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

    def clone(self) -> Self:
        """Create a shallow copy of this tool registry.

        Returns:
            A new instance of the same type with a copied tool registry.
            The shared context is not copied and must be set separately.
        """
        new = type(self)()
        new._registry.tools = self._registry.tools.copy()
        return new


class RealtimeTools(Tools):
    """Tool registry for the OpenAI Realtime API.

    Extends [`Tools`][rtvoice.tools.Tools] with schema serialisation in the
    format expected by the Realtime API. Used internally by `RealtimeAgent` —
    pass a plain [`Tools`][rtvoice.tools.Tools] instance to the agent rather
    than constructing this directly.
    """

    def get_tool_schema(self) -> list[FunctionTool]:
        """Return all registered tools serialised as Realtime API function schemas.

        Returns:
            List of [`FunctionTool`][rtvoice.realtime.schemas.FunctionTool] objects
            ready to be sent in a session update.
        """
        return self._registry.get_tool_schema()


class AgentTools(Tools):
    """Tool registry for non-realtime (text) agents such as `SupervisorAgent`.

    Extends [`Tools`][rtvoice.tools.Tools] with schema serialisation in the
    OpenAI Chat Completions `tools` format. Used internally — pass a plain
    [`Tools`][rtvoice.tools.Tools] instance to the agent rather than
    constructing this directly.
    """

    def get_json_tool_schema(self) -> list[dict]:
        """Return all registered tools serialised as Chat Completions tool schemas.

        Returns:
            List of dicts in the `{"type": "function", "function": {...}}` format
            expected by the Chat Completions API.
        """
        return [
            {
                "type": "function",
                "function": tool.model_dump(exclude={"type"}, exclude_none=True),
            }
            for tool in self._registry.get_tool_schema()
        ]
