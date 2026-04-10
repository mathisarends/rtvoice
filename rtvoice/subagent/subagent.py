import asyncio
import json
import logging
from typing import Annotated

from typing_extensions import Doc

from rtvoice.llm import (
    AssistantMessage,
    BaseChatModel,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from rtvoice.mcp import MCPServer
from rtvoice.shared.decorators import timed
from rtvoice.subagent.channel import CancellationToken
from rtvoice.subagent.views import (
    ClarifySignal,
    DoneSignal,
    SubAgentResult,
)
from rtvoice.tools import Tools
from rtvoice.tools.di import ToolContext

logger = logging.getLogger(__name__)


class SubAgent[T]:
    """Agentic sub-agent that can be delegated tasks from a `RealtimeAgent`.

    Runs an LLM-driven tool-calling loop to complete a given task, with built-in
    support for clarification questions, MCP server integration, and handoff
    from a parent voice agent.

    The agent exposes two special tools to the LLM automatically:

    - **done** — signals task completion and returns the final result.
    - **clarify** — asks the user a question and blocks until they answer.

    ```python
    agent = SubAgent(
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
            Tools | None,
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
        context: Annotated[
            T | None, Doc("Shared context object forwarded to all tool handlers.")
        ] = None,
    ) -> None:
        self.name = name.replace(" ", "_")
        self.description = description
        self._instructions = instructions
        self._llm = llm
        self._tools = Tools()
        if tools:
            self._tools.merge(tools)

        self._mcp_servers = mcp_servers or []
        self._max_iterations = max_iterations
        self.handoff_instructions = handoff_instructions
        self.result_instructions = result_instructions
        self.holding_instruction = holding_instruction

        self._cancellation: CancellationToken | None = None
        self._mcp_ready = asyncio.Event()

        self._tools.set_context(ToolContext(context=context))

        self._register_done_tool()
        self._register_clarify_tool()

    def attach_cancellation(self, token: CancellationToken) -> None:
        self._cancellation = token

    def _register_done_tool(self) -> None:
        @self._tools.action(
            "Signal that the task is complete and return the final result to the user. "
            "Only call this once you have gathered all necessary information or took the appropriate action."
        )
        def done(
            result: Annotated[str, "The final answer or result to return to the user."],
        ) -> DoneSignal:
            return DoneSignal(result)

    def _register_clarify_tool(self) -> None:
        @self._tools.action(
            "Ask the user a clarifying question when essential information is missing. "
            "Use sparingly – only when you cannot proceed without the answer. "
            "Calling this tool immediately returns control to the user; "
            "you will be called again once they answer."
        )
        def clarify(
            question: Annotated[str, "The question to ask the user."],
        ) -> ClarifySignal:
            return ClarifySignal(question)

    @timed()
    async def prewarm(
        self,
    ) -> None:
        if not self._mcp_ready.is_set():
            await self._connect_mcp_servers()

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
        SubAgentResult,
        Doc(
            "Final result including the message, success flag, and executed tool calls. "
            "If `clarification_needed` is set the caller must re-invoke `resume()` with "
            "the user's answer and the returned `resume_history` and `clarify_call_id`."
        ),
    ]:
        """Start a fresh tool-calling loop for the given task."""
        await self.prewarm()
        messages = self._build_messages(task=task, context=context)
        return await self._loop(messages)

    @timed()
    async def resume(
        self,
        clarification_answer: Annotated[
            str,
            Doc("The user's answer to the previous clarification question."),
        ],
        resume_history: Annotated[
            list[Message],
            Doc(
                "Message history from a previous run that was interrupted by a "
                "clarification request."
            ),
        ],
        clarify_call_id: Annotated[
            str,
            Doc(
                "Tool call ID of the previous `clarify` invocation. "
                "Used to construct a valid tool-result message so the LLM sees a "
                "correct message history on resume."
            ),
        ],
    ) -> Annotated[
        SubAgentResult,
        Doc(
            "Final result including the message, success flag, and executed tool calls. "
            "If `clarification_needed` is set the caller must re-invoke `resume()` again."
        ),
    ]:
        """Resume a previously interrupted run after the user answered a clarification question."""
        messages = list(resume_history)
        messages.append(
            ToolResultMessage(
                tool_call_id=clarify_call_id,
                content=clarification_answer,
            )
        )
        return await self._loop(messages)

    async def _loop(self, messages: list[Message]) -> SubAgentResult:
        tool_schema = self._tools.get_json_tool_schema()
        executed_tool_calls: list[ToolCall] = []

        try:
            for _ in range(self._max_iterations):
                if self._cancellation and self._cancellation.is_cancelled:
                    return SubAgentResult(
                        message="Task was cancelled by the user.",
                        success=False,
                        tool_calls=executed_tool_calls,
                    )

                response = await self._llm.invoke(messages, tools=tool_schema)

                if not response.tool_calls:
                    return SubAgentResult(
                        message=response.completion,
                        tool_calls=executed_tool_calls,
                    )

                messages.append(
                    AssistantMessage(
                        content=response.completion,
                        tool_calls=response.tool_calls,
                    )
                )

                for tool_call in response.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    logger.debug("Executing tool call: '%s'", tool_name)
                    result = await self._tools.execute(tool_name, tool_args)

                    match result:
                        case DoneSignal(result=msg):
                            logger.debug("Tool 'done' called with result: %s", msg)
                            return SubAgentResult(
                                success=True,
                                message=msg,
                                tool_calls=executed_tool_calls,
                                suppress_realtime_response=self._should_suppress_realtime_response(
                                    executed_tool_calls
                                ),
                            )
                        case ClarifySignal(question=question):
                            logger.debug(
                                "Tool 'clarify' called with question: %s", question
                            )
                            return SubAgentResult(
                                message="",
                                success=False,
                                tool_calls=executed_tool_calls,
                                clarification_needed=question,
                                resume_history=list(messages),
                                clarify_call_id=tool_call.id,
                            )

                    executed_tool_calls.append(tool_call)
                    messages.append(
                        ToolResultMessage(
                            tool_call_id=tool_call.id, content=str(result)
                        )
                    )

            return SubAgentResult(
                message="Max iterations reached without a final answer.",
                success=False,
                tool_calls=executed_tool_calls,
            )
        finally:
            self._cancellation = None

    def _build_messages(self, task: str, context: str | None) -> list[Message]:
        system_content = self._build_system_prompt()
        messages = [SystemMessage(content=system_content)]
        if context:
            messages.append(
                UserMessage(
                    content=f"<conversation_history>\n{context}\n</conversation_history>"
                )
            )
        messages.append(UserMessage(content=f"<task>\n{task}\n</task>"))
        return messages

    def _build_system_prompt(self) -> str:
        return self._instructions

    def _should_suppress_realtime_response(
        self, executed_tool_calls: list[ToolCall]
    ) -> bool:
        if not executed_tool_calls:
            return False

        last_call = executed_tool_calls[-1]
        last_tool = self._tools.get(last_call.function.name)

        return bool(last_tool and getattr(last_tool, "suppress_response", False))
