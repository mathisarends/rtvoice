import asyncio
import logging
from pathlib import Path
from typing import Annotated, Self

from llmify import BaseChatModel, SystemMessage, ToolResultMessage, UserMessage

from rtvoice.mcp import MCPServer
from rtvoice.subagents.skills import SkillRegistry
from rtvoice.subagents.views import SubAgentDone, SubAgentResult, ToolCall
from rtvoice.tools import Tools

logger = logging.getLogger(__name__)


class SubAgent:
    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        llm: BaseChatModel | None = None,
        tools: Tools | None = None,
        mcp_servers: list[MCPServer] | None = None,
        max_iterations: int = 10,
        skills_dir: str | Path | None = None,
        handoff_instructions: str | None = None,
        result_instructions: str | None = None,
        fire_and_forget: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self._instructions = instructions
        self._llm = llm
        self._tools = tools or Tools()
        self._mcp_servers = mcp_servers or []
        self._max_iterations = max_iterations
        self._skills = SkillRegistry(Path(skills_dir)) if skills_dir else None
        self.handoff_instructions = handoff_instructions
        self.result_instructions = result_instructions
        self.fire_and_forget = fire_and_forget

        self._mcp_ready = asyncio.Event()

        self._register_done_tool()
        if self._skills:
            self._register_skill_tools()

    def _register_done_tool(self) -> None:
        @self._tools.action(
            "Signal that the task is complete and return the final result to the user. "
            "Only call this once you have gathered all necessary information or took the appropriate action."
        )
        def done(
            result: Annotated[str, "The final answer or result to return to the user."],
        ) -> str:
            raise SubAgentDone(result)

    def _register_skill_tools(self) -> None:
        @self._tools.action(
            "Load the full instructions for a specific skill. "
            "Call this before starting a task if a relevant skill is available."
        )
        def load_skill(
            skill_name: Annotated[
                str, "The name of the skill to load (from the available skills list)."
            ],
        ) -> str:
            return self._skills.load(skill_name)

        @self._tools.action(
            "Load an additional resource file from a skill directory. "
            "Use this when the skill instructions reference a specific file."
        )
        def load_skill_resource(
            skill_name: Annotated[str, "The name of the skill."],
            resource_path: Annotated[
                str, "Relative path to the resource file within the skill directory."
            ],
        ) -> str:
            return self._skills.load_resource(skill_name, resource_path)

    def _build_system_prompt(self) -> str:
        if not self._skills or self._skills.is_empty():
            return self._instructions
        return f"{self._instructions}\n\n{self._skills.as_prompt_section()}"

    async def prepare(self) -> Self:
        await self._connect_mcp_servers()
        return self

    async def run(self, task: str, context: str | None = None) -> SubAgentResult:
        await self.prepare()

        tool_schema = self._tools.get_json_tool_schema()
        messages = [
            SystemMessage(self._build_system_prompt()),
        ]

        if context:
            messages.append(
                UserMessage(
                    f"<conversation_history>\n{context}\n</conversation_history>"
                )
            )

        messages.append(UserMessage(f"<task>\n{task}\n</task>"))
        executed_tool_calls: list[ToolCall] = []

        for _ in range(self._max_iterations):
            response = await self._llm.invoke(messages, tools=tool_schema)

            if not response.has_tool_calls:
                return SubAgentResult(
                    message=response.content, tool_calls=executed_tool_calls
                )

            messages.append(response.to_message())

            for tool_call in response.tool_calls:
                try:
                    result = await self._tools.execute(tool_call.name, tool_call.tool)
                except SubAgentDone as done:
                    return SubAgentResult(
                        success=True,
                        message=done.result,
                        tool_calls=executed_tool_calls,
                    )

                executed_tool_calls.append(
                    ToolCall(
                        name=tool_call.name,
                        arguments=tool_call.tool,
                        result=str(result),
                    )
                )

                messages.append(
                    ToolResultMessage(tool_call_id=tool_call.id, content=str(result))
                )

        return SubAgentResult(
            message="Max iterations reached without a final answer.",
            success=False,
            tool_calls=executed_tool_calls,
        )

    async def _setup_server(self, server: MCPServer) -> None:
        await server.connect()
        tools = await server.list_tools()
        for tool in tools:
            self._tools.register_mcp(tool, server)

    async def _connect_mcp_servers(self) -> None:
        if self._mcp_ready.is_set():
            return

        if not self._mcp_servers:
            self._mcp_ready.set()
            return

        results = await asyncio.gather(
            *[self._setup_server(s) for s in self._mcp_servers],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error("SubAgent MCP server failed: %s", result)

        self._mcp_ready.set()
