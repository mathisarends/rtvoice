import abc
import asyncio
import json
import logging
from typing import Annotated, Any

from pydantic import BaseModel
from typing_extensions import Doc

from rtvoice.mcp.views import (
    ClientInfo,
    InitializeParams,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    MCPToolDefinition,
    MCPToolsListResult,
)
from rtvoice.realtime.schemas import (
    FunctionParameterProperty,
    FunctionParameters,
    FunctionTool,
)

logger = logging.getLogger(__name__)


class MCPServer(abc.ABC):
    @abc.abstractmethod
    async def connect(self): ...

    @abc.abstractmethod
    async def cleanup(self): ...

    @abc.abstractmethod
    async def list_tools(self) -> list[FunctionTool]: ...

    @abc.abstractmethod
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None
    ) -> dict: ...

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.cleanup()


class MCPServerStdio(MCPServer):
    """MCP server implementation communicating over stdio (JSON-RPC 2.0).

    Spawns a subprocess and communicates via stdin/stdout using the
    Model Context Protocol. Tools are discovered via `list_tools()` and
    invoked via `call_tool()`.

    Example:
        ```python
        server = MCPServerStdio(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        async with server:
            tools = await server.list_tools()
        ```
    """

    def __init__(
        self,
        command: Annotated[str, Doc("Executable to spawn as the MCP server process.")],
        args: Annotated[
            list[str] | None,
            Doc("Arguments passed to the command."),
        ] = None,
        env: Annotated[
            dict[str, str] | None,
            Doc(
                "Environment variables for the subprocess. Inherits the current environment when `None`."
            ),
        ] = None,
        cache_tools_list: Annotated[
            bool,
            Doc("Cache the tools list after the first call to `list_tools()`."),
        ] = True,
        allowed_tools: Annotated[
            list[str] | None,
            Doc(
                "Whitelist of tool names to expose. All tools are exposed when `None`."
            ),
        ] = None,
    ):
        self._command = command
        self._args = args or []
        self._env = env
        self._cache_tools_list = cache_tools_list
        self._allowed_tools: set[str] | None = (
            set(allowed_tools) if allowed_tools is not None else None
        )
        self._process: asyncio.subprocess.Process | None = None
        self._msg_id = 0
        self._tools_cache: list[FunctionTool] | None = None

    async def connect(self) -> None:
        """Spawn the server process and complete the MCP handshake.

        If a process is already running, it is terminated first.
        Sends `initialize` and `notifications/initialized` to complete the protocol handshake.
        """
        if self._process is not None:
            await self.cleanup()

        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )

        await self._request(
            "initialize",
            InitializeParams(
                clientInfo=ClientInfo(name="rtvoice", version="0.1.0")
            ).model_dump(),
        )
        await self._notify("notifications/initialized")

    async def list_tools(self) -> list[FunctionTool]:
        """Fetch and return all tools exposed by the server.

        Results are cached after the first call if `cache_tools_list` is enabled.
        Applies `allowed_tools` filtering if configured.
        """
        if self._cache_tools_list and self._tools_cache is not None:
            return self._tools_cache

        result = await self._request("tools/list", {})
        tools_result = MCPToolsListResult.model_validate(result)
        tools = [self._parse_tool(t) for t in tools_result.tools]

        if self._allowed_tools is not None:
            tools = [t for t in tools if t.name in self._allowed_tools]

        self._tools_cache = tools
        return tools

    async def call_tool(
        self,
        tool_name: Annotated[str, Doc("Name of the tool to invoke.")],
        arguments: Annotated[
            dict[str, Any] | None,
            Doc("Arguments passed to the tool. Defaults to an empty dict if `None`."),
        ] = None,
    ) -> Annotated[dict, Doc("Raw result returned by the MCP server.")]:
        """Invoke a named tool on the server and return its result."""
        return await self._request(
            "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
        )

    async def cleanup(self) -> None:
        """Terminate the server process and clear cached state.

        Idempotent — safe to call even if the process is not running.
        """
        if self._process:
            self._process.terminate()
            await self._process.wait()
            self._process = None
            self._tools_cache = None

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_tool(tool: MCPToolDefinition) -> FunctionTool:
        properties = {
            name: FunctionParameterProperty.model_validate(prop)
            for name, prop in tool.inputSchema.properties.items()
        }
        return FunctionTool(
            name=tool.name,
            description=tool.description,
            parameters=FunctionParameters(
                properties=properties,
                required=tool.inputSchema.required,
            ),
        )

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def _send(self, message: BaseModel) -> None:
        assert self._process and self._process.stdin
        self._process.stdin.write((message.model_dump_json() + "\n").encode())
        await self._process.stdin.drain()

    async def _recv(self) -> JsonRpcResponse:
        assert self._process and self._process.stdout
        while True:
            line = await self._process.stdout.readline()
            if not line:
                stderr_output = b""
                if self._process.stderr:
                    stderr_output = await self._process.stderr.read()
                raise RuntimeError(
                    f"MCP server process exited unexpectedly.\nSTDERR: {stderr_output.decode()}"
                )
            line = line.strip()
            if not line:
                continue
            try:
                return JsonRpcResponse.model_validate_json(line)
            except (json.JSONDecodeError, ValueError):
                logger.debug("MCP server non-JSON stdout: %s", line.decode())
                continue

    async def _request(self, method: str, params: dict) -> dict:
        msg_id = self._next_id()
        await self._send(JsonRpcRequest(id=msg_id, method=method, params=params))
        while True:
            response = await self._recv()
            if response.id == msg_id:
                return response.unwrap()

    async def _notify(self, method: str, params: dict | None = None) -> None:
        await self._send(JsonRpcNotification(method=method, params=params))
