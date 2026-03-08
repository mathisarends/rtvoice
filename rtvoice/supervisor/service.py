import asyncio
import json
import logging
from typing import Annotated, Any, Self

from llmify import (
    BaseChatModel,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pydantic import BaseModel
from typing_extensions import Doc

from rtvoice.events.bus import EventBus
from rtvoice.mcp import MCPServer
from rtvoice.shared.decorators import timed
from rtvoice.supervisor.channel import SupervisorChannel
from rtvoice.supervisor.views import (
    SupervisorAgentDone,
    SupervisorAgentResult,
)
from rtvoice.tools import SupervisorTools
from rtvoice.tools.views import SpecialToolParameters

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Agentic sub-agent that can be delegated tasks from a `RealtimeAgent`.

    Runs an LLM-driven tool-calling loop to complete a given task, with built-in
    support for clarification questions, MCP server integration, and handoff
    from a parent voice agent.

    The agent exposes three special tools to the LLM automatically:

    - **done** — signals task completion and returns the final result.
    - **clarify** — asks the user a question and blocks until they answer.
    - **send_status** — sends a brief progress update to the user during long operations.

    Example:
        ```python
        agent = SupervisorAgent(
            name="calendar_agent",
            description="Manages the user's calendar.",
            instructions="You are a calendar assistant ...",
            llm=OpenAIChat(model="gpt-4o"),
        )
        result = await agent.run("Schedule a meeting with Alice tomorrow at 3pm.")
        ```
    """

    def __init__(
        self,
        name: Annotated[
            str,
            Doc(
                "Unique identifier for this agent. Used as the tool name when "
                "registered as a handoff target in a `RealtimeAgent` "
                "(spaces are replaced with underscores)."
            ),
        ],
        description: Annotated[
            str,
            Doc(
                "Short description shown to the parent LLM so it knows when "
                "to delegate tasks to this agent."
            ),
        ],
        instructions: Annotated[
            str,
            Doc("System prompt defining this agent's capabilities and behavior."),
        ],
        llm: Annotated[
            BaseChatModel | None,
            Doc(
                "LLM backend for the tool-calling loop. Must support tool/function calling."
            ),
        ] = None,
        tools: Annotated[
            SupervisorTools | None,
            Doc("Pre-registered tools available to the agent during its run loop."),
        ] = None,
        mcp_servers: Annotated[
            list[MCPServer] | None,
            Doc(
                "MCP servers connected during `prewarm()`. Their tools are registered automatically."
            ),
        ] = None,
        max_iterations: Annotated[
            int,
            Doc(
                "Maximum number of LLM invocations before the loop aborts. "
                "Guards against infinite tool-calling cycles."
            ),
        ] = 10,
        handoff_instructions: Annotated[
            str | None,
            Doc(
                "Extra instructions appended to the handoff tool description "
                "shown to the parent `RealtimeAgent`'s LLM."
            ),
        ] = None,
        result_instructions: Annotated[
            str | None,
            Doc(
                "Instructions for how the parent agent should present the result "
                "returned by this agent to the user."
            ),
        ] = None,
        holding_instruction: Annotated[
            str | None,
            Doc(
                "Message the parent agent says to the user while this agent is "
                "working (e.g. *'One moment, checking your calendar…'*)."
            ),
        ] = None,
    ) -> None:
        self.name = name
        self.description = description
        self._instructions = instructions
        self._llm = llm
        self._tools = SupervisorTools()
        if tools:
            self._tools._registry.tools = tools._registry.tools.copy()
        self._mcp_servers = mcp_servers or []
        self._max_iterations = max_iterations
        self.handoff_instructions = handoff_instructions
        self.result_instructions = result_instructions
        self.holding_instruction = holding_instruction

        self._event_bus: EventBus | None = None
        self._channel: SupervisorChannel | None = None
        self._mcp_ready = asyncio.Event()

        self._register_done_tool()
        self._register_clarify_tool()

    def _inject(self, *, event_bus: EventBus, context: Any = None) -> None:
        self._event_bus = event_bus
        self._tools.set_context(
            SpecialToolParameters(event_bus=event_bus, context=context)
        )

    def _attach_channel(self, channel: SupervisorChannel) -> None:
        """Called by ToolCallingWatchdog at the start of each run."""
        self._channel = channel

    def _register_done_tool(self) -> None:
        @self._tools.action(
            "Signal that the task is complete and return the final result to the user. "
            "Only call this once you have gathered all necessary information or took the appropriate action."
        )
        def done(
            result: Annotated[str, "The final answer or result to return to the user."],
        ) -> str:
            raise SupervisorAgentDone(result)

    def _register_clarify_tool(self) -> None:
        @self._tools.action(
            "Ask the user a clarifying question when essential information is missing. "
            "Use sparingly – only when you cannot proceed without the answer."
        )
        async def clarify(
            question: Annotated[str, "The question to ask the user."],
        ) -> str:
            if self._channel is None:
                raise RuntimeError(
                    f"SupervisorAgent '{self.name}' has no channel. "
                    "Ensure it is registered via RealtimeAgent."
                )
            return await self._channel.ask_user(question)

    @timed()
    async def prewarm(
        self,
    ) -> Annotated[Self, Doc("Returns `self` for optional chaining.")]:
        """Prewarm MCP connections before `run()`.

        Safe to call multiple times — a no-op if servers are already connected.
        """
        if not self._mcp_ready.is_set():
            await self._connect_mcp_servers()
        return self

    async def _connect_mcp_servers(self) -> None:
        if not self._mcp_servers:
            self._mcp_ready.set()
            return

        results = await asyncio.gather(
            *[self._connect_server(s) for s in self._mcp_servers],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error("MCP server connection failed: %s", result)

        self._mcp_ready.set()

    async def _connect_server(self, server: MCPServer) -> None:
        await server.connect()
        tools = await server.list_tools()
        for tool in tools:
            self._tools.register_mcp(tool, server)
        logger.info("MCP server connected: %d tools loaded", len(tools))

    @timed()
    async def run(
        self,
        task: Annotated[str, Doc("The task or question to complete.")],
        context: Annotated[
            str | None,
            Doc(
                "Optional conversation history from the parent voice session, "
                "injected as a `<conversation_history>` block so the agent has "
                "full context without re-asking the user."
            ),
        ] = None,
    ) -> Annotated[
        SupervisorAgentResult,
        Doc(
            "Final result including the message, success flag, and executed tool calls."
        ),
    ]:
        """Run the tool-calling loop until the task is complete or `max_iterations` is reached.

        Calls `prewarm()` automatically if not already done. Terminates when the
        LLM calls the `done` tool, when it responds without tool calls, or when
        `max_iterations` is exhausted.
        """
        await self.prewarm()

        messages = self._build_messages(task=task, context=context)

        tool_schema = self._tools.get_json_tool_schema()
        executed_tool_calls: list[ToolCall] = []

        try:
            for _ in range(self._max_iterations):
                if self._channel and self._channel.is_cancelled:
                    return SupervisorAgentResult(
                        message="Task was cancelled by the user.",
                        success=False,
                        tool_calls=executed_tool_calls,
                    )

                response = await self._llm.invoke(messages, tools=tool_schema)

                if not response.has_tool_calls:
                    return SupervisorAgentResult(
                        message=response.content,
                        tool_calls=executed_tool_calls,
                    )

                messages.append(response.to_message())

                for tool_call in response.tool_calls:
                    result = await self._execute_tool_call(
                        tool_call, executed_tool_calls, messages
                    )
                    if isinstance(result, SupervisorAgentResult):
                        return result

            return SupervisorAgentResult(
                message="Max iterations reached without a final answer.",
                success=False,
                tool_calls=executed_tool_calls,
            )
        finally:
            if self._channel:
                self._channel.close()
                self._channel = None

    def _build_messages(self, task: str, context: str | None) -> list[Message]:
        messages = [SystemMessage(self._instructions)]
        if context:
            messages.append(
                UserMessage(
                    f"<conversation_history>\n{context}\n</conversation_history>"
                )
            )
        messages.append(UserMessage(f"<task>\n{task}\n</task>"))
        return messages

    async def _execute_tool_call(
        self,
        tool_call: ToolCall,
        executed_tool_calls: list[ToolCall],
        messages: list,
    ) -> SupervisorAgentResult | None:
        await self._send_tool_status(tool_call)

        try:
            result = await self._tools.execute(tool_call.name, tool_call.tool)
        except SupervisorAgentDone as done:
            return SupervisorAgentResult(
                success=True,
                message=done.result,
                tool_calls=executed_tool_calls,
            )

        executed_tool_calls.append(
            ToolCall(id=tool_call.id, name=tool_call.name, tool=tool_call.tool)
        )
        messages.append(
            ToolResultMessage(tool_call_id=tool_call.id, content=str(result))
        )
        return None

    async def _send_tool_status(self, tool_call: ToolCall) -> None:
        if not self._channel or tool_call.name in ("done", "clarify"):
            return

        if isinstance(tool_call.tool, BaseModel):
            args_str = tool_call.tool.model_dump_json(exclude_none=True)
        elif isinstance(tool_call.tool, dict):
            args_str = json.dumps(tool_call.tool, ensure_ascii=False)
        else:
            args_str = str(tool_call.tool)

        await self._channel.send_status(f"{tool_call.name}({args_str})")
