import asyncio
import json
import logging
from typing import Annotated

from rtvoice.llm import (
    AssistantMessage,
    BaseChatModel,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from rtvoice.mcp import MCPServer
from rtvoice.shared.decorators import timed
from rtvoice.subagent.views import (
    AgentClarificationNeeded,
    AgentDone,
    ClarifySignal,
    DoneSignal,
    ProgressCallback,
    ProgressSignal,
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
    - **report_progress** — sends an intermediate status update to the user without interrupting the loop.

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
        name: str,
        description: str,
        instructions: str,
        llm: BaseChatModel | None = None,
        tools: Tools | None = None,
        mcp_servers: list[MCPServer] | None = None,
        max_iterations: int = 10,
        handoff_instructions: str | None = None,
        result_instructions: str | None = None,
        holding_instruction: str | None = None,
        context: T | None = None,
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

        self._mcp_ready = asyncio.Event()

        self._tools.set_context(ToolContext(context=context))

        self.on_progress: ProgressCallback | None = None

        self._register_done_tool()
        self._register_clarify_tool()
        self._register_progress_tool()

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
            "Use sparingly - only when you cannot proceed without the answer. "
            "Calling this tool immediately returns control to the user; "
            "you will be called again once they answer."
        )
        def clarify(
            question: Annotated[str, "The question to ask the user."],
        ) -> ClarifySignal:
            return ClarifySignal(question)

    def _register_progress_tool(self) -> None:
        @self._tools.action(
            "Report an intermediate progress update to the user while working on a long-running task. "
            "Use this to keep the user informed without blocking - the loop continues immediately after."
        )
        def report_progress(
            message: Annotated[str, "A short status update for the user."],
        ) -> ProgressSignal:
            return ProgressSignal(message)

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
        task: str,
        context: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> SubAgentResult:
        """Start a fresh tool-calling loop for the given task."""
        self._on_progress = on_progress or self.on_progress
        await self.prewarm()
        messages = self._build_messages(task=task, context=context)
        return await self._loop(messages)

    @timed()
    async def resume(
        self,
        clarification_answer: str,
        resume_history: list[Message],
        clarify_call_id: str,
        on_progress: ProgressCallback | None = None,
    ) -> SubAgentResult:
        """Resume a previously interrupted run after the user answered a clarification question."""
        self._on_progress = on_progress or self.on_progress
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

        for _ in range(self._max_iterations):
            response = await self._llm.invoke(messages, tools=tool_schema)

            if not response.tool_calls:
                return AgentDone(message=response.completion)

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
                        return AgentDone(message=msg)
                    case ClarifySignal(question=question):
                        logger.debug(
                            "Tool 'clarify' called with question: %s", question
                        )
                        return AgentClarificationNeeded(
                            question=question,
                            resume_history=list(messages),
                            clarify_call_id=tool_call.id,
                        )
                    case ProgressSignal(message=msg):
                        logger.debug("Progress update: %s", msg)
                        if self._on_progress:
                            await self._on_progress(msg)
                        messages.append(
                            ToolResultMessage(
                                tool_call_id=tool_call.id,
                                content="Progress noted, continue.",
                            )
                        )
                        continue

                content = str(result)
                steering = self._tools.get_steering(tool_name)
                if steering:
                    content = f"{content}\n\n<steering>{steering}</steering>"

                messages.append(
                    ToolResultMessage(tool_call_id=tool_call.id, content=content)
                )

        return AgentDone(message="Max iterations reached.", success=False)

    def _build_messages(self, task: str, context: str | None) -> list[Message]:
        messages = [SystemMessage(content=self._instructions)]
        if context:
            messages.append(
                UserMessage(
                    content=f"<conversation_history>\n{context}\n</conversation_history>"
                )
            )
        messages.append(UserMessage(content=f"<task>\n{task}\n</task>"))
        return messages
