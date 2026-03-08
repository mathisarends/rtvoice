from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
    FunctionCallItem,
)
from rtvoice.tools.registry.views import Tool
from rtvoice.watchdogs import ToolCallingWatchdog


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def websocket() -> AsyncMock:
    ws = AsyncMock()
    ws.send = AsyncMock()
    return ws


@pytest.fixture
def tools() -> MagicMock:
    t = MagicMock()
    t.get = MagicMock(return_value=None)
    t.execute = AsyncMock(return_value="tool_result")
    return t


@pytest.fixture
def watchdog(
    event_bus: EventBus, tools: MagicMock, websocket: AsyncMock
) -> ToolCallingWatchdog:
    return ToolCallingWatchdog(event_bus, tools, websocket)


def make_function_call_item(
    name: str = "get_weather",
    call_id: str = "call_001",
    arguments: dict | None = None,
) -> FunctionCallItem:
    return FunctionCallItem(
        event_id="evt_001",
        call_id=call_id,
        item_id="item_001",
        output_index=0,
        response_id="resp_001",
        name=name,
        arguments=arguments or {},
    )


def make_immediate_tool(name: str = "get_weather") -> MagicMock:
    tool = MagicMock(spec=Tool)
    tool.name = name
    tool.result_instruction = None
    tool.holding_instruction = None
    return tool


class TestImmediateTool:
    @pytest.mark.asyncio
    async def test_immediate_tool_sends_function_call_output(
        self,
        event_bus: EventBus,
        watchdog: ToolCallingWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_immediate_tool()

        await event_bus.dispatch(make_function_call_item())

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ConversationItemCreateEvent in sent_types

    @pytest.mark.asyncio
    async def test_immediate_tool_sends_response_create_after_result(
        self,
        event_bus: EventBus,
        watchdog: ToolCallingWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_immediate_tool()

        await event_bus.dispatch(make_function_call_item())

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ConversationResponseCreateEvent in sent_types

    @pytest.mark.asyncio
    async def test_immediate_tool_executes_with_arguments(
        self,
        event_bus: EventBus,
        watchdog: ToolCallingWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_immediate_tool()
        call = make_function_call_item(arguments={"city": "Berlin"})

        await event_bus.dispatch(call)

        tools.execute.assert_called_once_with("get_weather", {"city": "Berlin"})


class TestUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown_tool_does_not_send_to_websocket(
        self,
        event_bus: EventBus,
        watchdog: ToolCallingWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = None

        await event_bus.dispatch(make_function_call_item(name="nonexistent_tool"))

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_tool_does_not_execute(
        self,
        event_bus: EventBus,
        watchdog: ToolCallingWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = None

        await event_bus.dispatch(make_function_call_item(name="nonexistent_tool"))

        tools.execute.assert_not_called()


class TestSupervisorToolSkipped:
    @pytest.fixture
    def watchdog_with_supervisor(
        self, event_bus: EventBus, tools: MagicMock, websocket: AsyncMock
    ) -> ToolCallingWatchdog:
        return ToolCallingWatchdog(
            event_bus, tools, websocket, supervisor_tool_names={"slow_job"}
        )

    @pytest.mark.asyncio
    async def test_supervisor_tool_is_not_executed(
        self,
        event_bus: EventBus,
        watchdog_with_supervisor: ToolCallingWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_immediate_tool(name="slow_job")

        await event_bus.dispatch(make_function_call_item(name="slow_job"))

        tools.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_supervisor_tool_does_not_send_to_websocket(
        self,
        event_bus: EventBus,
        watchdog_with_supervisor: ToolCallingWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_immediate_tool(name="slow_job")

        await event_bus.dispatch(make_function_call_item(name="slow_job"))

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_supervisor_tool_still_executes(
        self,
        event_bus: EventBus,
        watchdog_with_supervisor: ToolCallingWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_immediate_tool(name="get_weather")

        await event_bus.dispatch(make_function_call_item(name="get_weather"))

        tools.execute.assert_called_once_with("get_weather", {})
