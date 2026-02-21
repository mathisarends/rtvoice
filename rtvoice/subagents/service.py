from llmify import BaseChatModel, SystemMessage, ToolResultMessage, UserMessage

from rtvoice.mcp import MCPServer
from rtvoice.tools import Tools


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
    ):
        self.name = name
        self.description = description
        self._instructions = instructions
        self._tools = tools or Tools()
        self._mcp_servers = mcp_servers or []
        self._llm = llm
        self._max_iterations = max_iterations

    async def run(self, task: str) -> str:
        await self._connect_mcp_servers()

        tool_schema = self._tools.get_json_tool_schema()

        messages = [
            SystemMessage(self._instructions),
            UserMessage(task),
        ]

        for _ in range(self._max_iterations):
            response = await self._llm.invoke(messages, tools=tool_schema)

            if not response.has_tool_calls:
                return response.content

            messages.append(response.to_message())

            for tool_call in response.tool_calls:
                result = await self._tools.execute(tool_call.name, tool_call.tool)
                messages.append(
                    ToolResultMessage(tool_call_id=tool_call.id, content=str(result))
                )

        return "Max iterations reached without completing the task."

    async def _connect_mcp_servers(self) -> None:
        for server in self._mcp_servers:
            await server.connect()
            tools = await server.list_tools()
            for tool in tools:
                self._tools.register_mcp(tool, server)
