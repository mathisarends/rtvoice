from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from rtvoice.events.bus import EventBus
from rtvoice.tools import Tools
from rtvoice.tools.di import Inject, ToolContext


class EventSearchParams(BaseModel):
    query: str
    date: str


class EventCreateParams(BaseModel):
    date: str
    attendees: str


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
    async def test_injects_via_inject_marker(self, tools: Tools) -> None:
        injected_bus = EventBus()
        context = ToolContext(event_bus=injected_bus)
        tools.set_context(context)

        received = {}

        @tools.action(description="Uses event bus via Inject")
        async def handler(bus: Inject[EventBus]) -> None:
            received["bus"] = bus

        await tools.execute("handler", {})

        assert received["bus"] is injected_bus

    @pytest.mark.asyncio
    async def test_name_without_inject_marker_is_not_injected(
        self, tools: Tools
    ) -> None:
        injected_bus = EventBus()
        context = ToolContext(event_bus=injected_bus)
        tools.set_context(context)

        @tools.action(description="Uses event bus by name only")
        async def handler(event_bus: EventBus) -> None: ...

        with pytest.raises(ValueError, match="event_bus"):
            await tools.execute("handler", {})

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

    @pytest.mark.asyncio
    async def test_execute_works_without_context_for_regular_parameters(
        self, tools: Tools
    ) -> None:
        @tools.action(description="Uppercase text")
        async def uppercase(text: str) -> str:
            return text.upper()

        result = await tools.execute("uppercase", {"text": "hello"})

        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_missing_context_with_injected_param_raises_value_error(
        self, tools: Tools
    ) -> None:
        @tools.action(description="Needs injected event bus")
        async def handler(bus: Inject[EventBus]) -> None:
            _ = bus

        with pytest.raises(ValueError, match="bus"):
            await tools.execute("handler", {})


class TestSetContext:
    def test_context_is_updated(self, tools: Tools) -> None:
        new_context = ToolContext()
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


class TestJsonToolSchema:
    def test_get_json_tool_schema_returns_list(self) -> None:
        tools = Tools()

        @tools.action(description="A tool")
        def my_tool(name: str) -> None: ...

        schema = tools.get_json_tool_schema()

        assert len(schema) == 1

    def test_get_json_tool_schema_has_function_type(self) -> None:
        tools = Tools()

        @tools.action(description="A tool")
        def my_tool(name: str) -> None: ...

        schema = tools.get_json_tool_schema()

        assert schema[0]["type"] == "function"

    def test_get_json_tool_schema_contains_function_key(self) -> None:
        tools = Tools()

        @tools.action(description="A tool")
        def my_tool(name: str) -> None: ...

        schema = tools.get_json_tool_schema()

        assert "function" in schema[0]

    def test_get_json_tool_schema_empty_for_no_tools(self) -> None:
        tools = Tools()

        assert tools.get_json_tool_schema() == []


class TestSubAgentTools:
    def test_tool_format_status_formats_template(self) -> None:
        agent = Tools()

        @agent.action(
            description="Search events",
            param_model=EventSearchParams,
            status="Suche nach '{query}' im Kalender...",
        )
        def search_events(params: EventSearchParams) -> list:
            return []

        tool = agent.get("search_events")
        assert tool is not None
        status = tool.format_status({"query": "Zahnarzt", "date": "2026-03-15"})

        assert status == "Suche nach 'Zahnarzt' im Kalender..."

    def test_tool_format_status_returns_none_without_status_template(self) -> None:
        agent = Tools()

        @agent.action(description="Search events")
        def search_events(query: str) -> list:
            return []

        tool = agent.get("search_events")
        assert tool is not None
        status = tool.format_status({"query": "Dentist"})

        assert status is None

    def test_tool_format_status_falls_back_to_template_on_missing_key(self) -> None:
        agent = Tools()

        @agent.action(
            description="Create event",
            param_model=EventCreateParams,
            status="Erstelle Termin am {date} mit {attendees}...",
        )
        def create_event(params: EventCreateParams) -> str:
            return "ok"

        tool = agent.get("create_event")
        assert tool is not None
        status = tool.format_status({"date": "Montag"})

        assert status == "Erstelle Termin am {date} mit {attendees}..."
