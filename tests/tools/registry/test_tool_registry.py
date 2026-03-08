from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


class TestActionDecorator:
    def test_registers_tool_by_function_name(self, registry: ToolRegistry) -> None:
        @registry.action(description="Say hello")
        def greet(name: str) -> None: ...

        assert registry.get("greet") is not None

    def test_registers_tool_with_custom_name(self, registry: ToolRegistry) -> None:
        @registry.action(description="Say hello", name="say_hello")
        def greet(name: str) -> None: ...

        assert registry.get("say_hello") is not None

    def test_stores_description(self, registry: ToolRegistry) -> None:
        @registry.action(description="Says hello to the user")
        def greet(name: str) -> None: ...

        assert registry.get("greet").description == "Says hello to the user"

    def test_stores_result_instruction(self, registry: ToolRegistry) -> None:
        @registry.action(
            description="Do something", result_instruction="Tell the user it's done"
        )
        def do_something() -> None: ...

        assert (
            registry.get("do_something").result_instruction == "Tell the user it's done"
        )

    def test_stores_holding_instruction(self, registry: ToolRegistry) -> None:
        @registry.action(description="Long task", holding_instruction="Please wait...")
        def long_task() -> None: ...

        assert registry.get("long_task").holding_instruction == "Please wait..."

    def test_decorator_returns_original_function(self, registry: ToolRegistry) -> None:
        def greet(name: str) -> str:
            return f"Hello {name}"

        decorated = registry.action(description="Greet")(greet)

        assert decorated is greet

    def test_duplicate_name_raises_value_error(self, registry: ToolRegistry) -> None:
        @registry.action(description="First")
        def greet() -> None: ...

        with pytest.raises(ValueError, match="greet"):

            @registry.action(description="Second")
            def greet() -> None: ...


class TestGet:
    def test_returns_none_for_unknown_tool(self, registry: ToolRegistry) -> None:
        assert registry.get("nonexistent") is None

    def test_returns_tool_after_registration(self, registry: ToolRegistry) -> None:
        @registry.action(description="A tool")
        def my_tool() -> None: ...

        assert registry.get("my_tool") is not None

    def test_returns_correct_tool_by_name(self, registry: ToolRegistry) -> None:
        @registry.action(description="First tool")
        def tool_a() -> None: ...

        @registry.action(description="Second tool")
        def tool_b() -> None: ...

        assert registry.get("tool_a").name == "tool_a"
        assert registry.get("tool_b").name == "tool_b"


class TestGetToolSchema:
    def test_empty_registry_returns_empty_list(self, registry: ToolRegistry) -> None:
        assert registry.get_tool_schema() == []

    def test_returns_one_schema_per_tool(self, registry: ToolRegistry) -> None:
        @registry.action(description="Tool A")
        def tool_a() -> None: ...

        @registry.action(description="Tool B")
        def tool_b() -> None: ...

        assert len(registry.get_tool_schema()) == 2

    def test_schema_contains_tool_name(self, registry: ToolRegistry) -> None:
        @registry.action(description="A tool")
        def my_tool() -> None: ...

        schema = registry.get_tool_schema()[0]

        assert schema.name == "my_tool"


class TestRegisterMcp:
    def test_registers_mcp_tool_by_name(self, registry: ToolRegistry) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = "Does something via MCP"
        mcp_tool.parameters = MagicMock()
        server = MagicMock()

        registry.register_mcp(mcp_tool, server)

        assert registry.get("mcp_action") is not None

    def test_mcp_tool_stores_description(self, registry: ToolRegistry) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = "Does something via MCP"
        mcp_tool.parameters = MagicMock()
        server = MagicMock()

        registry.register_mcp(mcp_tool, server)

        assert registry.get("mcp_action").description == "Does something via MCP"

    def test_mcp_tool_none_description_defaults_to_empty_string(
        self, registry: ToolRegistry
    ) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = None
        mcp_tool.parameters = MagicMock()
        server = MagicMock()

        registry.register_mcp(mcp_tool, server)

        assert registry.get("mcp_action").description == ""

    @pytest.mark.asyncio
    async def test_mcp_handler_calls_server_with_kwargs(
        self, registry: ToolRegistry
    ) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = "Action"
        mcp_tool.parameters = MagicMock()
        server = MagicMock()
        server.call_tool = AsyncMock(return_value="result")

        registry.register_mcp(mcp_tool, server)

        tool = registry.get("mcp_action")
        await tool.function(key="value")

        server.call_tool.assert_called_once_with("mcp_action", {"key": "value"})

    @pytest.mark.asyncio
    async def test_mcp_handler_passes_none_for_empty_kwargs(
        self, registry: ToolRegistry
    ) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = "Action"
        mcp_tool.parameters = MagicMock()
        server = MagicMock()
        server.call_tool = AsyncMock(return_value="result")

        registry.register_mcp(mcp_tool, server)

        tool = registry.get("mcp_action")
        await tool.function()

        server.call_tool.assert_called_once_with("mcp_action", None)

    def test_duplicate_mcp_tool_raises_value_error(
        self, registry: ToolRegistry
    ) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = "Action"
        mcp_tool.parameters = MagicMock()
        server = MagicMock()

        registry.register_mcp(mcp_tool, server)

        with pytest.raises(ValueError, match="mcp_action"):
            registry.register_mcp(mcp_tool, server)
