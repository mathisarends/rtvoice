import asyncio
import logging
from pathlib import Path
from typing import Annotated, Self

from llmify import (
    BaseChatModel,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from typing_extensions import Doc

from rtvoice.mcp import MCPServer
from rtvoice.shared.decorators import timed
from rtvoice.skills import Skill, SkillRegistry
from rtvoice.subagent.channel import SubAgentChannel
from rtvoice.subagent.views import (
    ClarifySignal,
    DoneSignal,
    SubAgentResult,
)
from rtvoice.tools import SubAgentTools
from rtvoice.tools.views import SpecialToolParameters

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
            SubAgentTools | None,
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
        skills: Annotated[
            list[Skill | Path | str] | None,
            Doc(
                "Agent skills to make available. Each entry is either a Skill instance "
                "or a path to a skill directory containing a SKILL.md file. "
                "Skills are injected into the system prompt; with dynamic_skills=True "
                "the agent loads them on-demand via a 'load_skill' tool."
            ),
        ] = None,
        dynamic_skills: Annotated[
            bool,
            Doc(
                "If True, only a skill index is injected into the system prompt and the "
                "agent explicitly calls 'load_skill' to pull in full instructions. "
                "Reduces token usage for large skill sets."
            ),
        ] = False,
    ) -> None:
        self.name = name.replace(" ", "_")
        self.description = description
        self._instructions = instructions
        self._llm = llm
        self._tools = SubAgentTools()
        if tools:
            self._tools.merge(tools)

        self._mcp_servers = mcp_servers or []
        self._max_iterations = max_iterations
        self.handoff_instructions = handoff_instructions
        self.result_instructions = result_instructions
        self.holding_instruction = holding_instruction

        self._channel: SubAgentChannel | None = None
        self._mcp_ready = asyncio.Event()
        self._skill_registry = SkillRegistry()

        if skills:
            self._skill_registry.add(*skills)

        self._dynamic_skills = dynamic_skills

        self._tools.set_context(SpecialToolParameters(context=context))

        if self._skill_registry.skills:
            self._register_skill_tools()

        self._register_done_tool()
        self._register_clarify_tool()

    def _attach_channel(self, channel: SubAgentChannel) -> None:
        """Called by ToolCallingWatchdog at the start of each run."""
        self._channel = channel

    def attach_channel(self, channel: SubAgentChannel) -> None:
        self._attach_channel(channel)

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

    def _register_skill_tools(self) -> None:
        registry = self._skill_registry

        @self._tools.action(
            "Load the full instructions for a specific agent skill. "
            "Call this before attempting tasks that require specialized knowledge. "
            f"Available skills: {registry.available_names()}"
        )
        def load_skill(
            skill_name: Annotated[str, "Name of the skill to load."],
        ) -> str:
            skill = registry.get(skill_name)
            if skill is None:
                available = ", ".join(registry.available_names())
                return f"Skill '{skill_name}' not found. Available: {available}"
            return f"<skill name='{skill.name}'>\n{skill.instructions}\n</skill>"

        @self._tools.action("List all available agent skills with their descriptions.")
        def list_skills() -> str:
            if not registry.skills:
                return "No skills available."
            return registry.format_index()

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
        clarification_answer: Annotated[
            str | None,
            Doc(
                "Answer to a previous clarification question. "
                "Must be provided together with `resume_history` and `clarify_call_id`."
            ),
        ] = None,
        clarify_call_id: Annotated[
            str | None,
            Doc(
                "Tool call ID of the previous `clarify` invocation. "
                "Used to construct a valid tool-result message so the LLM sees a "
                "correct message history on resume."
            ),
        ] = None,
        resume_history: Annotated[
            list | None,
            Doc(
                "Message history from a previous run that was interrupted by a "
                "clarification request. When provided, the loop resumes from this "
                "state rather than starting fresh."
            ),
        ] = None,
    ) -> Annotated[
        SubAgentResult,
        Doc(
            "Final result including the message, success flag, and executed tool calls. "
            "If `clarification_needed` is set the caller must re-invoke `run()` with "
            "the user's answer and the returned `resume_history` and `clarify_call_id`."
        ),
    ]:
        """Run the tool-calling loop until the task is complete or `max_iterations` is reached."""
        await self.prewarm()

        if resume_history is not None and clarification_answer is not None:
            messages = resume_history
            messages.append(
                ToolResultMessage(
                    tool_call_id=clarify_call_id or "",
                    content=clarification_answer,
                )
            )
        else:
            messages = self._build_messages(task=task, context=context)

        tool_schema = self._tools.get_json_tool_schema()
        executed_tool_calls: list[ToolCall] = []

        try:
            for _ in range(self._max_iterations):
                if self._channel and self._channel.is_cancelled:
                    return SubAgentResult(
                        message="Task was cancelled by the user.",
                        success=False,
                        tool_calls=executed_tool_calls,
                    )

                response = await self._llm.invoke(messages, tools=tool_schema)

                if not response.has_tool_calls:
                    return SubAgentResult(
                        message=response.content,
                        tool_calls=executed_tool_calls,
                    )

                messages.append(response.to_message())

                for tool_call in response.tool_calls:
                    early_return = await self._execute_tool_call(
                        tool_call, executed_tool_calls, messages
                    )
                    if early_return is not None:
                        return early_return

            return SubAgentResult(
                message="Max iterations reached without a final answer.",
                success=False,
                tool_calls=executed_tool_calls,
            )
        finally:
            if self._channel:
                self._channel.close()
                self._channel = None

    def _build_messages(self, task: str, context: str | None) -> list[Message]:
        system_content = self._build_system_prompt()
        messages = [SystemMessage(system_content)]
        if context:
            messages.append(
                UserMessage(
                    f"<conversation_history>\n{context}\n</conversation_history>"
                )
            )
        messages.append(UserMessage(f"<task>\n{task}\n</task>"))
        return messages

    def _build_system_prompt(self) -> str:
        parts = [self._instructions]

        if self._skill_registry.skills:
            if self._dynamic_skills:
                parts.append(
                    "\n\nYou have access to agent skills that provide specialized "
                    "instructions and capabilities. Use 'list_skills' to see what's "
                    "available and 'load_skill' to load instructions before tackling "
                    "a task that requires that expertise.\n"
                )
                parts.append(self._skill_registry.format_index())
            else:
                parts.append(self._skill_registry.format_all_instructions())

        return "\n".join(parts)

    async def _execute_tool_call(
        self,
        tool_call: ToolCall,
        executed_tool_calls: list[ToolCall],
        messages: list,
    ) -> SubAgentResult | None:
        logger.debug("Executing tool call: '%s'", tool_call.name)
        await self._send_tool_status(tool_call)

        result = await self._tools.execute(tool_call.name, tool_call.tool)

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
                logger.debug("Tool 'clarify' called with question: %s", question)
                return SubAgentResult(
                    message="",
                    success=False,
                    tool_calls=executed_tool_calls,
                    clarification_needed=question,
                    resume_history=list(messages),
                    clarify_call_id=tool_call.id,
                )

        executed_tool_calls.append(
            ToolCall(id=tool_call.id, name=tool_call.name, tool=tool_call.tool)
        )
        messages.append(
            ToolResultMessage(tool_call_id=tool_call.id, content=str(result))
        )
        return None

    def _should_suppress_realtime_response(
        self, executed_tool_calls: list[ToolCall]
    ) -> bool:
        if not executed_tool_calls:
            return False

        last_call = executed_tool_calls[-1]
        last_tool = self._tools.get(last_call.name)

        return bool(last_tool and getattr(last_tool, "suppress_response", False))

    async def _send_tool_status(self, tool_call: ToolCall) -> None:
        if not self._channel or self._is_default_registered_tool(tool_call.name):
            return

        tool = self._tools.get(tool_call.name)
        if tool is None:
            return

        status = tool.format_status(tool_call.tool)
        if status is None:
            return

        logger.debug("Sending status update via channel: %s", status)
        self._channel.buffer_status(status)

    def _is_default_registered_tool(self, tool_name: str) -> bool:
        return tool_name in ("done", "clarify")
