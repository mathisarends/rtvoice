import asyncio
from typing import Annotated

from llmify import BaseChatModel, SystemMessage, ToolResultMessage, UserMessage

from rtvoice.mcp import MCPServer
from rtvoice.subagents.views import SubAgentDone
from rtvoice.tools import Tools
from rtvoice.views import ActionResult


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
        pending_message: str | None = None,
        handoff_instructions: str | None = None,
    ):
        self.name = name
        self.description = description
        self._instructions = instructions
        self._tools = tools or Tools()
        self._mcp_servers = mcp_servers or []
        self._llm = llm
        self._max_iterations = max_iterations
        self.pending_message = pending_message
        self.handoff_instructions = handoff_instructions

        self._register_done_tool()

    def _register_done_tool(self) -> None:
        @self._tools.action(
            "Signal that the task is complete and return the final result to the user. "
            "Only call this once you have gathered all necessary information."
        )
        def done(
            result: Annotated[str, "The final answer or result to return to the user."],
        ) -> str:
            raise SubAgentDone(result)

    async def run(self, task: str) -> ActionResult:
        await self._connect_mcp_servers()

        tool_schema = self._tools.get_json_tool_schema()
        messages = [
            SystemMessage(self._instructions),
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
        async def _setup_server(server: MCPServer) -> None:
            await server.connect()
            tools = await server.list_tools()
            for tool in tools:
                self._tools.register_mcp(tool, server)

        await asyncio.gather(*[_setup_server(s) for s in self._mcp_servers])
