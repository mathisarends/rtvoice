import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
    FunctionCallItem,
    RealtimeResponseObject,
    RealtimeServerEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
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


def make_supervisor_agent() -> MagicMock:
    agent = MagicMock()
    agent._attach_channel.side_effect = lambda channel: channel.close()
    return agent


def make_response_created(response_id: str = "resp_hold") -> ResponseCreatedEvent:
    return ResponseCreatedEvent(
        type=RealtimeServerEvent.RESPONSE_CREATED,
        event_id="evt_002",
        response=RealtimeResponseObject(id=response_id),
    )


def make_response_done(response_id: str = "resp_hold") -> ResponseDoneEvent:
    return ResponseDoneEvent(
        type=RealtimeServerEvent.RESPONSE_DONE,
        event_id="evt_003",
        response=RealtimeResponseObject(id=response_id),
    )


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


class TestSupervisorTool:
    @pytest.fixture(autouse=True)
    def register_supervisor(self, watchdog: ToolCallingWatchdog) -> None:
        watchdog.register_supervisor("slow_job", make_supervisor_agent())

    @pytest.mark.asyncio
    async def test_long_running_sends_holding_response_immediately(
        self,
        event_bus: EventBus,
        watchdog: ToolCallingWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_immediate_tool(name="slow_job")

        await event_bus.dispatch(make_function_call_item(name="slow_job"))

        websocket.send.assert_called_once()
        sent = websocket.send.call_args[0][0]
        assert isinstance(sent, ConversationResponseCreateEvent)

    @pytest.mark.asyncio
    async def test_long_running_delivers_result_after_holding_done(
        self,
        event_bus: EventBus,
        watchdog: ToolCallingWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_immediate_tool(name="slow_job")
        tools.execute = AsyncMock(return_value="job_done")

        await event_bus.dispatch(
            make_function_call_item(name="slow_job", call_id="call_lr")
        )
        await event_bus.dispatch(make_response_created("resp_hold"))
        await event_bus.dispatch(make_response_done("resp_hold"))
        await asyncio.sleep(0.05)

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ConversationItemCreateEvent in sent_types

    @pytest.mark.asyncio
    async def test_duplicate_long_running_sends_already_in_progress(
        self,
        event_bus: EventBus,
        watchdog: ToolCallingWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_immediate_tool(name="slow_job")

        _block = asyncio.Event()

        async def blocking_execute(name: str, arguments: dict):
            await _block.wait()

        tools.execute = blocking_execute

        await event_bus.dispatch(
            make_function_call_item(name="slow_job", call_id="call_1")
        )
        await event_bus.dispatch(
            make_function_call_item(name="slow_job", call_id="call_2")
        )

        all_sent = [c.args[0] for c in websocket.send.call_args_list]
        fn_outputs = [s for s in all_sent if isinstance(s, ConversationItemCreateEvent)]
        assert any(
            "already in progress" in s.item.output.lower()
            for s in fn_outputs
            if hasattr(s.item, "output")
        )


class TestResponseTracking:
    @pytest.fixture(autouse=True)
    def register_supervisor(self, watchdog: ToolCallingWatchdog) -> None:
        watchdog.register_supervisor("slow_job", make_supervisor_agent())

    @pytest.mark.asyncio
    async def test_response_created_assigns_holding_response_id(
        self,
        event_bus: EventBus,
        watchdog: ToolCallingWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_immediate_tool(name="slow_job")
        _block = asyncio.Event()

        async def blocking_execute(name: str, arguments: dict):
            await _block.wait()

        tools.execute = blocking_execute

        await event_bus.dispatch(make_function_call_item(name="slow_job"))
        await event_bus.dispatch(make_response_created("resp_hold"))

        assert watchdog._pending[0].holding_response_id == "resp_hold"

    @pytest.mark.asyncio
    async def test_response_done_removes_pending_entry(
        self,
        event_bus: EventBus,
        watchdog: ToolCallingWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_immediate_tool(name="slow_job")
        tools.execute = AsyncMock(return_value="done")

        await event_bus.dispatch(make_function_call_item(name="slow_job"))
        await event_bus.dispatch(make_response_created("resp_hold"))
        await event_bus.dispatch(make_response_done("resp_hold"))
        await asyncio.sleep(0.05)

        assert len(watchdog._pending) == 0
