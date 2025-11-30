import asyncio
import json
from typing import Any

from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.tools.mcp.models import MCPServerConfig, MCPServerType, MCPToolMetadata


class MCPServer(LoggingMixin):
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._tools: dict[str, MCPToolMetadata] = {}
        self._request_id = 0

    async def start(self) -> None:
        if self.config.type != MCPServerType.STDIO:
            raise ValueError(f"Only stdio servers supported, got {self.config.type}")

        self.config.validate_config()

        self.logger.info(
            "Starting MCP server '%s': %s %s",
            self.config.name,
            self.config.command,
            " ".join(self.config.args),
        )

        try:
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.config.env or None,
            )

            await self._initialize()
            await self._discover_tools()

            self.logger.info(
                "MCP server '%s' started with %d tools",
                self.config.name,
                len(self._tools),
            )

        except Exception as e:
            self.logger.exception("Failed to start MCP server '%s'", self.config.name)
            await self.stop()
            raise RuntimeError(
                f"Failed to start MCP server '{self.config.name}': {e}"
            ) from e

    async def stop(self) -> None:
        if not self._process:
            return

        self.logger.info("Stopping MCP server '%s'", self.config.name)

        try:
            self._process.terminate()
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except TimeoutError:
            self.logger.warning(
                "MCP server '%s' did not terminate, killing", self.config.name
            )
            self._process.kill()
            await self._process.wait()
        except Exception as e:
            self.logger.exception(
                "Error stopping MCP server '%s'", self.config.name, exc_info=e
            )

        self._process = None
        self._tools.clear()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name not in self._tools:
            raise ValueError(
                f"Tool '{tool_name}' not found on server '{self.config.name}'"
            )

        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        response = await self._send_request(request)

        if "error" in response:
            error = response["error"]
            raise RuntimeError(
                f"MCP tool error: {error.get('message', 'Unknown error')}"
            )

        return response.get("result")

    async def _initialize(self) -> None:
        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "rtvoice", "version": "1.0.0"},
            },
        }

        response = await self._send_request(request)

        if "error" in response:
            raise RuntimeError(f"Failed to initialize: {response['error']}")

        # Send initialized notification
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}

        await self._send_notification(notification)

    async def _discover_tools(self) -> None:
        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/list",
        }

        response = await self._send_request(request)

        if "error" in response:
            raise RuntimeError(f"Failed to list tools: {response['error']}")

        tools = response.get("result", {}).get("tools", [])

        for tool in tools:
            metadata = MCPToolMetadata(
                server_name=self.config.name,
                tool_name=tool["name"],
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {}),
            )
            self._tools[tool["name"]] = metadata

        self.logger.debug(
            "Discovered tools from '%s': %s",
            self.config.name,
            list(self._tools.keys()),
        )

    async def _send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("MCP server not running")

        request_line = json.dumps(request) + "\n"
        self._process.stdin.write(request_line.encode())
        await self._process.stdin.drain()

        try:
            response_line = await asyncio.wait_for(
                self._process.stdout.readline(), timeout=self.config.timeout
            )
        except TimeoutError as e:
            raise RuntimeError(
                f"Timeout waiting for response from MCP server '{self.config.name}'"
            ) from e

        if not response_line:
            raise RuntimeError(f"MCP server '{self.config.name}' closed connection")

        return json.loads(response_line.decode())

    async def _send_notification(self, notification: dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("MCP server not running")

        notification_line = json.dumps(notification) + "\n"
        self._process.stdin.write(notification_line.encode())
        await self._process.stdin.drain()

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    @property
    def tools(self) -> dict[str, MCPToolMetadata]:
        return self._tools.copy()


class MCPServerManager(LoggingMixin):
    def __init__(self, configs: list[MCPServerConfig] | None = None):
        self._configs = configs or []
        self._servers: dict[str, MCPServer] = {}

    async def start_all(self) -> None:
        for config in self._configs:
            try:
                server = MCPServer(config)
                await server.start()
                self._servers[config.name] = server
            except Exception as e:
                self.logger.exception(
                    "Failed to start MCP server '%s', continuing with others",
                    config.name,
                    exc_info=e,
                )

    async def stop_all(self) -> None:
        tasks = [server.stop() for server in self._servers.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._servers.clear()

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        server = self._servers.get(server_name)
        if not server:
            raise ValueError(f"MCP server '{server_name}' not found")

        return await server.call_tool(tool_name, arguments)

    def get_all_tools(self) -> dict[str, MCPToolMetadata]:
        all_tools = {}
        for server in self._servers.values():
            for tool_name, metadata in server.tools.items():
                # Prefix with server name to avoid conflicts
                qualified_name = f"{metadata.server_name}.{tool_name}"
                all_tools[qualified_name] = metadata
        return all_tools
