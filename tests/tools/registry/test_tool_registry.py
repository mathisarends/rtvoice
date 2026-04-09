from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.tools import Tools


@pytest.fixture
def tools() -> Tools:
    return Tools()


class TestActionDecorator:
    def test_registers_tool_by_function_name(self, tools: Tools) -> None:
        @tools.action(description="Say hello")
        def greet(name: str) -> None: ...

        assert tools.get("greet") is not None

    def test_registers_tool_with_custom_name(self, tools: Tools) -> None:
        @tools.action(description="Say hello", name="say_hello")
        def greet(name: str) -> None: ...

        assert tools.get("say_hello") is not None

    def test_stores_description(self, tools: Tools) -> None:
        @tools.action(description="Says hello to the user")
        def greet(name: str) -> None: ...

        assert tools.get("greet").description == "Says hello to the user"

    def test_stores_result_instruction(self, tools: Tools) -> None:
        @tools.action(
            description="Do something", result_instruction="Tell the user it's done"
        )
        def do_something() -> None: ...

        assert tools.get("do_something").result_instruction == "Tell the user it's done"

    def test_stores_holding_instruction(self, tools: Tools) -> None:
        @tools.action(description="Long task", holding_instruction="Please wait...")
        def long_task() -> None: ...

        tool = tools.get("long_task")
        assert tool is not None
        assert tool.holding_instruction == "Please wait..."

    def test_decorator_returns_original_function(self, tools: Tools) -> None:
        def greet(name: str) -> str:
            return f"Hello {name}"

        decorated = tools.action(description="Greet")(greet)

        assert decorated is greet

    def test_stores_status_template(self, tools: Tools) -> None:
        @tools.action(
            description="Draft email",
            status="Entwerfe eine Email an {recipient}...",
        )
        def draft_email(recipient: str, subject: str, body: str) -> str:
            return "ok"

        tool = tools.get("draft_email")
        assert tool is not None
        assert tool.status == "Entwerfe eine Email an {recipient}..."

    def test_invalid_status_template_raises_value_error(self, tools: Tools) -> None:
        with pytest.raises(ValueError, match="unknown placeholders"):

            @tools.action(
                description="Draft email",
                status="Entwerfe eine Email an {empfaenger}...",
            )
            def draft_email(recipient: str, subject: str, body: str) -> str:
                return "ok"

    def test_stores_suppress_response_flag(self, tools: Tools) -> None:
        @tools.action(
            description="Silent task",
            suppress_response=True,
        )
        def silent_task(topic: str) -> str:
            return topic

        tool = tools.get("silent_task")
        assert tool is not None
        assert tool.suppress_response is True

    def test_duplicate_name_raises_value_error(self, tools: Tools) -> None:
        @tools.action(description="First")
        def greet() -> None: ...

        with pytest.raises(ValueError, match="greet"):

            @tools.action(description="Second")
            def greet() -> None: ...


class TestGet:
    def test_returns_none_for_unknown_tool(self, tools: Tools) -> None:
        assert tools.get("nonexistent") is None

    def test_returns_tool_after_registration(self, tools: Tools) -> None:
        @tools.action(description="A tool")
        def my_tool() -> None: ...

        assert tools.get("my_tool") is not None

    def test_returns_correct_tool_by_name(self, tools: Tools) -> None:
        @tools.action(description="First tool")
        def tool_a() -> None: ...

        @tools.action(description="Second tool")
        def tool_b() -> None: ...

        assert tools.get("tool_a").name == "tool_a"
        assert tools.get("tool_b").name == "tool_b"


class TestGetToolSchema:
    def test_empty_returns_empty_list(self, tools: Tools) -> None:
        assert tools.get_tool_schema() == []

    def test_returns_one_schema_per_tool(self, tools: Tools) -> None:
        @tools.action(description="Tool A")
        def tool_a() -> None: ...

        @tools.action(description="Tool B")
        def tool_b() -> None: ...

        assert len(tools.get_tool_schema()) == 2

    def test_schema_contains_tool_name(self, tools: Tools) -> None:
        @tools.action(description="A tool")
        def my_tool() -> None: ...

        schema = tools.get_tool_schema()[0]

        assert schema.name == "my_tool"


class TestRegisterMcp:
    def test_registers_mcp_tool_by_name(self, tools: Tools) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = "Does something via MCP"
        mcp_tool.parameters = MagicMock()
        server = MagicMock()

        tools.register_mcp(mcp_tool, server)

        assert tools.get("mcp_action") is not None

    def test_mcp_tool_stores_description(self, tools: Tools) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = "Does something via MCP"
        mcp_tool.parameters = MagicMock()
        server = MagicMock()

        tools.register_mcp(mcp_tool, server)

        assert tools.get("mcp_action").description == "Does something via MCP"

    def test_mcp_tool_none_description_defaults_to_empty_string(
        self, tools: Tools
    ) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = None
        mcp_tool.parameters = MagicMock()
        server = MagicMock()

        tools.register_mcp(mcp_tool, server)

        assert tools.get("mcp_action").description == ""

    @pytest.mark.asyncio
    async def test_mcp_handler_calls_server_with_kwargs(self, tools: Tools) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = "Action"
        mcp_tool.parameters = MagicMock()
        server = MagicMock()
        server.call_tool = AsyncMock(return_value="result")

        tools.register_mcp(mcp_tool, server)

        tool = tools.get("mcp_action")
        await tool.function(key="value")

        server.call_tool.assert_called_once_with("mcp_action", {"key": "value"})

    @pytest.mark.asyncio
    async def test_mcp_handler_passes_none_for_empty_kwargs(self, tools: Tools) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = "Action"
        mcp_tool.parameters = MagicMock()
        server = MagicMock()
        server.call_tool = AsyncMock(return_value="result")

        tools.register_mcp(mcp_tool, server)

        tool = tools.get("mcp_action")
        await tool.function()

        server.call_tool.assert_called_once_with("mcp_action", None)

    def test_duplicate_mcp_tool_raises_value_error(self, tools: Tools) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = "Action"
        mcp_tool.parameters = MagicMock()
        server = MagicMock()

        tools.register_mcp(mcp_tool, server)

        with pytest.raises(ValueError, match="mcp_action"):
            tools.register_mcp(mcp_tool, server)
