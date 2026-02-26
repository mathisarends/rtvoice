import abc
import asyncio
import json
import logging
from typing import Any

from rtvoice.realtime.schemas import (
    FunctionParameterProperty,
    FunctionParameters,
    FunctionTool,
)

logger = logging.getLogger(__name__)


def _parse_tool(raw: dict) -> FunctionTool:
    input_schema = raw.get("inputSchema", {})

    raw_properties = input_schema.get("properties", {})
    properties = {
        name: FunctionParameterProperty.model_validate(prop)
        for name, prop in raw_properties.items()
    }

    parameters = FunctionParameters(
        properties=properties,
        required=input_schema.get("required", []),
    )

    return FunctionTool(
        name=raw["name"],
        description=raw.get("description"),
        parameters=parameters,
    )


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
    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict | None = None,
        cache_tools_list: bool = True,
        allowed_tools: list[str] | None = None,
    ):
        self.command = command
        self.args = args if args else []
        self.env = env
        self.cache_tools_list = cache_tools_list
        self._allowed_tools: set[str] | None = (
            set(allowed_tools) if allowed_tools is not None else None
        )

        self._process: asyncio.subprocess.Process | None = None
        self._msg_id = 0
        self._tools_cache: list[FunctionTool] | None = None

    async def connect(self):
        self._process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "jarvis", "version": "0.1.0"},
            },
        )
        await self._notify("notifications/initialized")

    async def list_tools(self) -> list[FunctionTool]:
        if self.cache_tools_list and self._tools_cache is not None:
            return self._tools_cache
        result = await self._request("tools/list", {})

        tools = [_parse_tool(t) for t in result.get("tools", [])]

        if self._allowed_tools is not None:
            tools = [t for t in tools if t.name in self._allowed_tools]

        self._tools_cache = tools
        return self._tools_cache

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict:
        return await self._request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments or {},
            },
        )

    async def cleanup(self):
        if self._process:
            self._process.terminate()
            await self._process.wait()
            self._process = None

    # ── JSON-RPC ────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def _send(self, message: dict):
        assert self._process and self._process.stdin
        self._process.stdin.write((json.dumps(message) + "\n").encode())
        await self._process.stdin.drain()

    async def _recv(self) -> dict:
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
                return json.loads(line)
            except json.JSONDecodeError:
                logger.debug("MCP server non-JSON stdout: %s", line.decode())
                continue

    async def _request(self, method: str, params: dict) -> dict:
        msg_id = self._next_id()
        await self._send(
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        )
        while True:
            response = await self._recv()
            if response.get("id") == msg_id:
                if "error" in response:
                    raise RuntimeError(f"MCP error: {response['error']}")
                return response.get("result", {})

    async def _notify(self, method: str, params: dict | None = None):
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})
