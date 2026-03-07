import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rtvoice.mcp.server import MCPServerStdio


def make_rpc_response(msg_id: int, result: dict) -> bytes:
    return (
        json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}) + "\n"
    ).encode()


def make_tools_response(msg_id: int, tools: list[dict]) -> bytes:
    return make_rpc_response(msg_id, {"tools": tools})


def make_process(stdout_lines: list[bytes]) -> MagicMock:
    process = MagicMock()
    process.stdin = AsyncMock()
    process.stdin.write = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdout = AsyncMock()
    process.stderr = AsyncMock()
    process.stderr.read = AsyncMock(return_value=b"")
    process.terminate = MagicMock()
    process.wait = AsyncMock()

    responses = iter([*stdout_lines, b""])
    process.stdout.readline = AsyncMock(side_effect=lambda: next(responses))

    return process


SAMPLE_TOOL = {
    "name": "read_file",
    "description": "Read a file",
    "inputSchema": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path"}},
        "required": ["path"],
    },
}

HANDSHAKE_RESPONSES = [
    make_rpc_response(1, {"protocolVersion": "2024-11-05", "capabilities": {}}),
]


@pytest.fixture
def server() -> MCPServerStdio:
    return MCPServerStdio(command="npx", args=["-y", "some-server"])


@pytest.fixture
def connected_server() -> MCPServerStdio:
    return MCPServerStdio(command="npx", args=["-y", "some-server"])


class TestConnect:
    @pytest.mark.asyncio
    async def test_spawns_subprocess(self, server: MCPServerStdio) -> None:
        process = make_process(HANDSHAKE_RESPONSES)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()

        assert server._process is process

    @pytest.mark.asyncio
    async def test_sends_initialize_request(self, server: MCPServerStdio) -> None:
        process = make_process(HANDSHAKE_RESPONSES)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()

        written = b"".join(call.args[0] for call in process.stdin.write.call_args_list)
        assert b'"initialize"' in written

    @pytest.mark.asyncio
    async def test_sends_initialized_notification(self, server: MCPServerStdio) -> None:
        process = make_process(HANDSHAKE_RESPONSES)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()

        written = b"".join(call.args[0] for call in process.stdin.write.call_args_list)
        assert b'"notifications/initialized"' in written

    @pytest.mark.asyncio
    async def test_terminates_existing_process_before_reconnecting(
        self, server: MCPServerStdio
    ) -> None:
        first_process = make_process(HANDSHAKE_RESPONSES)
        second_process = make_process(HANDSHAKE_RESPONSES)

        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=[first_process, second_process]),
        ):
            await server.connect()
            await server.connect()

        first_process.terminate.assert_called_once()


class TestCleanup:
    @pytest.mark.asyncio
    async def test_terminates_process(self, server: MCPServerStdio) -> None:
        process = make_process(HANDSHAKE_RESPONSES)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()
            await server.cleanup()

        process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_sets_process_to_none(self, server: MCPServerStdio) -> None:
        process = make_process(HANDSHAKE_RESPONSES)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()
            await server.cleanup()

        assert server._process is None

    @pytest.mark.asyncio
    async def test_clears_tools_cache(self, server: MCPServerStdio) -> None:
        process = make_process(
            [*HANDSHAKE_RESPONSES, make_tools_response(2, [SAMPLE_TOOL])]
        )

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()
            await server.list_tools()
            await server.cleanup()

        assert server._tools_cache is None

    @pytest.mark.asyncio
    async def test_cleanup_without_connect_does_not_raise(
        self, server: MCPServerStdio
    ) -> None:
        await server.cleanup()


class TestListTools:
    @pytest.mark.asyncio
    async def test_returns_parsed_tools(self, server: MCPServerStdio) -> None:
        process = make_process(
            [*HANDSHAKE_RESPONSES, make_tools_response(2, [SAMPLE_TOOL])]
        )

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()
            tools = await server.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "read_file"

    @pytest.mark.asyncio
    async def test_caches_tools_on_second_call(self, server: MCPServerStdio) -> None:
        process = make_process(
            [*HANDSHAKE_RESPONSES, make_tools_response(2, [SAMPLE_TOOL])]
        )

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()
            await server.list_tools()
            await server.list_tools()

        assert process.stdout.readline.call_count == len(HANDSHAKE_RESPONSES) + 1

    @pytest.mark.asyncio
    async def test_cache_disabled_refetches(self) -> None:
        server = MCPServerStdio(command="npx", cache_tools_list=False)
        process = make_process(
            [
                *HANDSHAKE_RESPONSES,
                make_tools_response(2, [SAMPLE_TOOL]),
                make_tools_response(3, [SAMPLE_TOOL]),
            ]
        )

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()
            await server.list_tools()
            await server.list_tools()

        assert process.stdout.readline.call_count == len(HANDSHAKE_RESPONSES) + 2

    @pytest.mark.asyncio
    async def test_filters_by_allowed_tools(self) -> None:
        server = MCPServerStdio(command="npx", allowed_tools=["read_file"])
        other_tool = {**SAMPLE_TOOL, "name": "write_file"}
        process = make_process(
            [*HANDSHAKE_RESPONSES, make_tools_response(2, [SAMPLE_TOOL, other_tool])]
        )

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()
            tools = await server.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "read_file"

    @pytest.mark.asyncio
    async def test_returns_all_tools_when_no_filter(
        self, server: MCPServerStdio
    ) -> None:
        other_tool = {**SAMPLE_TOOL, "name": "write_file"}
        process = make_process(
            [*HANDSHAKE_RESPONSES, make_tools_response(2, [SAMPLE_TOOL, other_tool])]
        )

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()
            tools = await server.list_tools()

        assert len(tools) == 2


class TestCallTool:
    @pytest.mark.asyncio
    async def test_sends_tools_call_request(self, server: MCPServerStdio) -> None:
        call_response = make_rpc_response(
            2, {"content": [{"type": "text", "text": "file content"}]}
        )
        process = make_process([*HANDSHAKE_RESPONSES, call_response])

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()
            await server.call_tool("read_file", {"path": "/tmp/test.txt"})

        written = b"".join(call.args[0] for call in process.stdin.write.call_args_list)
        assert b'"tools/call"' in written
        assert b'"read_file"' in written

    @pytest.mark.asyncio
    async def test_passes_none_arguments_as_empty_dict(
        self, server: MCPServerStdio
    ) -> None:
        call_response = make_rpc_response(2, {"content": []})
        process = make_process([*HANDSHAKE_RESPONSES, call_response])

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()
            await server.call_tool("ping", None)

        written = b"".join(call.args[0] for call in process.stdin.write.call_args_list)
        assert b'"arguments":{}' in written


class TestRecv:
    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_process_exits(
        self, server: MCPServerStdio
    ) -> None:
        process = make_process([*HANDSHAKE_RESPONSES, b""])

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()

            with pytest.raises(RuntimeError, match="exited unexpectedly"):
                await server.list_tools()

    @pytest.mark.asyncio
    async def test_skips_non_json_lines(self, server: MCPServerStdio) -> None:
        process = make_process(
            [
                *HANDSHAKE_RESPONSES,
                b"some debug output\n",
                make_tools_response(2, [SAMPLE_TOOL]),
            ]
        )

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await server.connect()
            tools = await server.list_tools()

        assert len(tools) == 1


class TestContextManager:
    @pytest.mark.asyncio
    async def test_connect_called_on_enter(self, server: MCPServerStdio) -> None:
        process = make_process(HANDSHAKE_RESPONSES)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            async with server:
                pass

        assert process.terminate.called

    @pytest.mark.asyncio
    async def test_cleanup_called_on_exit(self, server: MCPServerStdio) -> None:
        process = make_process(HANDSHAKE_RESPONSES)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            async with server:
                pass

        process.terminate.assert_called_once()
