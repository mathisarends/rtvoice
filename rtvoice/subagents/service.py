from llmify import BaseChatModel, SystemMessage, ToolResultMessage, UserMessage

from rtvoice.mcp import MCPServer
from rtvoice.tools import Tools
from rtvoice.views import ActionResult

# TODO: initial state loading time oder so (brauchen aber alle tools)
# gerne auch tools unterstützen die generatoren sind und schrittweise ergebnisse liefern (damit man auch weiß was hier schrittweise passiert für die beste ux)

# der soll hier ja einen multi call tool loop haben udn dann semantisch entscheiden ob er ein done tool aufrufen wollte oder nicht | momentan nimmt der ja immer nur ein
# Tool das ist ja nciht so wie das gewollt ist actually

# tool um den user zu informierne, done tool


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
    ):
        self.name = name
        self.description = description
        self._instructions = instructions
        self._tools = tools or Tools()
        self._mcp_servers = mcp_servers or []
        self._llm = llm
        self._max_iterations = max_iterations

        self.pending_message = pending_message

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
                return ActionResult(content=response.content)

            messages.append(response.to_message())

            for tool_call in response.tool_calls:
                result = await self._tools.execute(tool_call.name, tool_call.tool)
                messages.append(
                    ToolResultMessage(tool_call_id=tool_call.id, content=str(result))
                )

        return ActionResult(
            content="Max iterations reached without a final answer.",
            success=False,
        )

    async def _connect_mcp_servers(self) -> None:
        for server in self._mcp_servers:
            await server.connect()
            tools = await server.list_tools()
            for tool in tools:
                self._tools.register_mcp(tool, server)
