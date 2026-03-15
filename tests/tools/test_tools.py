from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.tools import SubAgentTools, Tools
from rtvoice.tools.views import SpecialToolParameters


@pytest.fixture
def tools() -> Tools:
    return Tools()


class TestAction:
    def test_registers_tool(self, tools: Tools) -> None:
        @tools.action(description="Do something")
        def my_tool() -> None: ...

        assert tools.get("my_tool") is not None

    def test_registers_with_custom_name(self, tools: Tools) -> None:
        @tools.action(description="Do something", name="custom")
        def my_tool() -> None: ...

        assert tools.get("custom") is not None

    def test_returns_original_function(self, tools: Tools) -> None:
        def my_tool() -> str:
            return "result"

        decorated = tools.action(description="Do something")(my_tool)

        assert decorated is my_tool

    def test_duplicate_name_raises_value_error(self, tools: Tools) -> None:
        @tools.action(description="First")
        def my_tool() -> None: ...

        with pytest.raises(ValueError, match="my_tool"):

            @tools.action(description="Second")
            def my_tool() -> None: ...


class TestGet:
    def test_returns_none_for_unknown_tool(self, tools: Tools) -> None:
        assert tools.get("nonexistent") is None

    def test_returns_registered_tool(self, tools: Tools) -> None:
        @tools.action(description="A tool")
        def my_tool() -> None: ...

        assert tools.get("my_tool") is not None


class TestExecute:
    @pytest.mark.asyncio
    async def test_executes_tool_with_arguments(self, tools: Tools) -> None:
        @tools.action(description="Add numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        result = await tools.execute("add", {"a": 1, "b": 2})

        assert result == 3

    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_tool(self, tools: Tools) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            await tools.execute("nonexistent", {})

    @pytest.mark.asyncio
    async def test_raises_value_error_for_missing_required_param(
        self, tools: Tools
    ) -> None:
        @tools.action(description="Needs param")
        async def greet(name: str) -> str:
            return f"Hello {name}"

        with pytest.raises(ValueError, match="name"):
            await tools.execute("greet", {})

    @pytest.mark.asyncio
    async def test_optional_param_uses_default_when_not_provided(
        self, tools: Tools
    ) -> None:
        @tools.action(description="Greet")
        async def greet(name: str = "World") -> str:
            return f"Hello {name}"

        result = await tools.execute("greet", {})

        assert result == "Hello World"


class TestPrepareArguments:
    @pytest.mark.asyncio
    async def test_injects_special_params_from_context(self, tools: Tools) -> None:
        injected_bus = EventBus()
        context = SpecialToolParameters(event_bus=injected_bus)
        tools.set_context(context)

        received = {}

        @tools.action(description="Uses event bus")
        async def handler(event_bus) -> None:
            received["event_bus"] = event_bus

        await tools.execute("handler", {})

        assert received["event_bus"] is injected_bus

    @pytest.mark.asyncio
    async def test_llm_arguments_take_precedence_over_injected(
        self, tools: Tools
    ) -> None:
        received = {}

        @tools.action(description="Takes name")
        async def handler(name: str) -> None:
            received["name"] = name

        await tools.execute("handler", {"name": "explicit"})

        assert received["name"] == "explicit"


class TestSetContext:
    def test_context_is_updated(self, tools: Tools) -> None:
        new_context = SpecialToolParameters()
        tools.set_context(new_context)

        assert tools._context is new_context


class TestClone:
    def test_clone_is_different_instance(self, tools: Tools) -> None:
        clone = tools.clone()

        assert clone is not tools

    def test_clone_shares_registered_tools(self, tools: Tools) -> None:
        @tools.action(description="A tool")
        def my_tool() -> None: ...

        clone = tools.clone()

        assert clone.get("my_tool") is not None

    def test_clone_registry_is_independent(self, tools: Tools) -> None:
        @tools.action(description="Original tool")
        def original() -> None: ...

        clone = tools.clone()

        @clone.action(description="Clone only")
        def clone_only() -> None: ...

        assert tools.get("clone_only") is None

    def test_clone_preserves_type(self) -> None:
        realtime = Tools()
        clone = realtime.clone()

        assert type(clone) is Tools


class TestRegisterMcp:
    def test_registers_mcp_tool(self, tools: Tools) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = "MCP tool"
        mcp_tool.parameters = MagicMock()
        server = MagicMock()

        tools.register_mcp(mcp_tool, server)

        assert tools.get("mcp_action") is not None

    @pytest.mark.asyncio
    async def test_mcp_tool_is_executable(self, tools: Tools) -> None:
        mcp_tool = MagicMock()
        mcp_tool.name = "mcp_action"
        mcp_tool.description = "MCP tool"
        mcp_tool.parameters = MagicMock()
        server = MagicMock()
        server.call_tool = AsyncMock(return_value="mcp_result")

        tools.register_mcp(mcp_tool, server)
        result = await tools.execute("mcp_action", {"key": "value"})

        assert result == "mcp_result"


class TestRealtimeTools:
    def test_get_tool_schema_returns_list(self) -> None:
        realtime = Tools()

        @realtime.action(description="A tool")
        def my_tool(name: str) -> None: ...

        schema = realtime.get_tool_schema()

        assert len(schema) == 1

    def test_get_tool_schema_empty_for_no_tools(self) -> None:
        realtime = Tools()

        assert realtime.get_tool_schema() == []

    def test_action_does_not_accept_status(self) -> None:
        realtime = Tools()

        with pytest.raises(TypeError, match="status"):
            realtime.action(description="A tool", status="Working...")


class TestSubAgentTools:
    def test_get_json_tool_schema_returns_list(self) -> None:
        agent = SubAgentTools()

        @agent.action(description="A tool")
        def my_tool(name: str) -> None: ...

        schema = agent.get_json_tool_schema()

        assert len(schema) == 1

    def test_get_json_tool_schema_has_function_type(self) -> None:
        agent = SubAgentTools()

        @agent.action(description="A tool")
        def my_tool(name: str) -> None: ...

        schema = agent.get_json_tool_schema()

        assert schema[0]["type"] == "function"

    def test_get_json_tool_schema_contains_function_key(self) -> None:
        agent = SubAgentTools()

        @agent.action(description="A tool")
        def my_tool(name: str) -> None: ...

        schema = agent.get_json_tool_schema()

        assert "function" in schema[0]

    def test_get_json_tool_schema_empty_for_no_tools(self) -> None:
        agent = SubAgentTools()

        assert agent.get_json_tool_schema() == []

    def test_action_does_not_accept_holding_instruction(self) -> None:
        agent = SubAgentTools()

        with pytest.raises(TypeError, match="holding_instruction"):
            agent.action(description="A tool", holding_instruction="Please wait")

    def test_tool_format_status_formats_template(self) -> None:
        agent = SubAgentTools()

        @agent.action(
            description="Search events",
            status="Suche nach '{query}' im Kalender...",
        )
        def search_events(query: str, date: str) -> list:
            return []

        tool = agent.get("search_events")
        assert tool is not None
        status = tool.format_status({"query": "Zahnarzt", "date": "2026-03-15"})

        assert status == "Suche nach 'Zahnarzt' im Kalender..."

    def test_tool_format_status_returns_none_without_status_template(self) -> None:
        agent = SubAgentTools()

        @agent.action(description="Search events")
        def search_events(query: str) -> list:
            return []

        tool = agent.get("search_events")
        assert tool is not None
        status = tool.format_status({"query": "Dentist"})

        assert status is None

    def test_tool_format_status_falls_back_to_template_on_missing_key(self) -> None:
        agent = SubAgentTools()

        @agent.action(
            description="Create event",
            status="Erstelle Termin am {date} mit {attendees}...",
        )
        def create_event(date: str, attendees: str) -> str:
            return "ok"

        tool = agent.get("create_event")
        assert tool is not None
        status = tool.format_status({"date": "Montag"})

        assert status == "Erstelle Termin am {date} mit {attendees}..."

    def test_action_stores_suppress_response_flag(self) -> None:
        agent = SubAgentTools()

        @agent.action(description="Silent op", suppress_response=True)
        def silent_op() -> str:
            return "ok"

        tool = agent.get("silent_op")
        assert tool is not None
        assert tool.suppress_response is True
