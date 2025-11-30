from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Self

from mcp import ClientSession, StdioServerParameters, stdio_client

from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.tools.mcp.converter import MCPToolConverter
from rtvoice.tools.models import FunctionTool

if TYPE_CHECKING:
    from rtvoice.tools.mcp.models import MCPServerConfig


class MCPServer(LoggingMixin):
    def __init__(self, config: "MCPServerConfig"):
        self.config = config
        self._session: ClientSession | None = None
        self._tools: list[FunctionTool] = []
        self._converter = MCPToolConverter()
        self._exit_stack = AsyncExitStack()  # ✅ Verwaltet alle Context Manager

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def start(self) -> None:
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
        )

        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        await self._session.initialize()

        self._tools = await self._converter.convert_from_session(
            self._session, self.config.name
        )

        self.logger.info(
            f"MCP server '{self.config.name}' started with {len(self._tools)} tools"
        )

    async def stop(self) -> None:
        # ✅ Schließt alle Context Manager in umgekehrter Reihenfolge
        await self._exit_stack.aclose()

        self.logger.info(f"MCP server '{self.config.name}' stopped")

    @property
    def tools(self) -> list[FunctionTool]:
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self._session:
            raise RuntimeError(f"MCP server '{self.config.name}' is not running")

        # Remove server prefix from tool name
        original_name = tool_name.replace(f"{self.config.name}__", "")

        result = await self._session.call_tool(original_name, arguments)

        if result.content:
            return str(
                result.content[0].text
                if hasattr(result.content[0], "text")
                else result.content[0]
            )

        return ""
