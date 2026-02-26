import asyncio
import logging
from pathlib import Path
from typing import Annotated, Self

from llmify import BaseChatModel, SystemMessage, ToolResultMessage, UserMessage

from rtvoice.mcp import MCPServer
from rtvoice.subagents.views import SubAgentDone
from rtvoice.tools import Tools
from rtvoice.views import ActionResult

logger = logging.getLogger(__name__)


class SubAgent:
    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        tools: Tools | None = None,
        mcp_servers: list[MCPServer] | None = None,
        llm: BaseChatModel | None = None,
        max_iterations: int = 10,
        handoff_instructions: str | None = None,
        result_instructions: str | None = None,
        fire_and_forget: bool = False,
        skills_dir: str | Path | None = None,
    ):
        self.name = name
        self.description = description
        self._instructions = instructions
        self._tools = tools or Tools()
        self._mcp_servers = mcp_servers or []
        self._llm = llm
        self._max_iterations = max_iterations
        self.handoff_instructions = handoff_instructions
        self.result_instructions = result_instructions
        self.fire_and_forget = fire_and_forget
        self._skills_dir = Path(skills_dir) if skills_dir else None

        self._mcp_ready = asyncio.Event()

        self._register_done_tool()
        if self._skills_dir:
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
        skills_index = self._build_skills_index()
        if not skills_index:
            return

        self._skills_index = skills_index

        @self._tools.action(
            "Load the full instructions for a specific skill. "
            "Call this before starting a task if a relevant skill is available."
        )
        def load_skill(
            skill_name: Annotated[
                str, "The name of the skill to load (from the available skills list)."
            ],
        ) -> str:
            skill_path = self._skills_dir / skill_name / "SKILL.md"
            if not skill_path.exists():
                return f"Skill '{skill_name}' not found."
            return skill_path.read_text()

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
            full_path = self._skills_dir / skill_name / resource_path
            if not full_path.resolve().is_relative_to(self._skills_dir.resolve()):
                return "Access denied: path outside skills directory."
            if not full_path.exists():
                return f"Resource '{resource_path}' not found in skill '{skill_name}'."
            return full_path.read_text()

    def _build_skills_index(self) -> dict[str, str]:
        index = {}
        if not self._skills_dir or not self._skills_dir.exists():
            return index

        for skill_md in self._skills_dir.glob("*/SKILL.md"):
            try:
                content = skill_md.read_text()
                name = skill_md.parent.name
                description = self._extract_description(content)
                index[name] = description
            except Exception as e:
                logger.warning("Could not index skill at %s: %s", skill_md, e)

        return index

    def _extract_description(self, skill_content: str) -> str:
        import re

        match = re.search(r"^description:\s*(.+)$", skill_content, re.MULTILINE)
        return match.group(1).strip() if match else "No description."

    def _build_system_prompt(self) -> str:
        base = self._instructions
        if (
            not self._skills_dir
            or not hasattr(self, "_skills_index")
            or not self._skills_index
        ):
            return base

        skills_list = "\n".join(
            f"- {name}: {desc}" for name, desc in self._skills_index.items()
        )
        return (
            f"{base}\n\n"
            f"## Available Skills\n"
            f"You have access to the following skills. "
            f"Call `load_skill` with the skill name before starting a relevant task.\n"
            f"{skills_list}"
        )

    async def prepare(self) -> Self:
        await self._connect_mcp_servers()
        return self

    async def run(self, task: str) -> ActionResult:
        await self.prepare()

        tool_schema = self._tools.get_json_tool_schema()
        messages = [
            SystemMessage(self._build_system_prompt()),
            UserMessage(task),
        ]

        for _ in range(self._max_iterations):
            response = await self._llm.invoke(messages, tools=tool_schema)

            if not response.has_tool_calls:
                return ActionResult(message=response.content)

            messages.append(response.to_message())

            for tool_call in response.tool_calls:
                try:
                    result = await self._tools.execute(tool_call.name, tool_call.tool)
                except SubAgentDone as done:
                    return ActionResult(success=True, message=done.result)

                messages.append(
                    ToolResultMessage(tool_call_id=tool_call.id, content=str(result))
                )

        return ActionResult(
            message="Max iterations reached without a final answer.",
            success=False,
        )

    async def _connect_mcp_servers(self) -> None:
        if self._mcp_ready.is_set():
            return

        if not self._mcp_servers:
            self._mcp_ready.set()
            return

        async def _setup_server(server: MCPServer) -> tuple[MCPServer, list]:
            await server.connect()
            tools = await server.list_tools()
            return server, tools

        results = await asyncio.gather(
            *[_setup_server(s) for s in self._mcp_servers],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error("SubAgent MCP server failed: %s", result)
                continue
            server, tools = result
            for tool in tools:
                self._tools.register_mcp(tool, server)

        self._mcp_ready.set()
