import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from transitbus import EventBus

from rtvoice.events.views import (
    ToolExecutedEvent,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
)
from rtvoice.handler import ToolCallExecutor
from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
    FunctionCallItem,
    RealtimeResponseObject,
    RealtimeServerEvent,
    ResponseDoneEvent,
)
from rtvoice.tools import ActionKind, ActionResult
from rtvoice.tools.views import Tool


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
    registry = MagicMock()
    registry.get = MagicMock(return_value=None)
    registry.execute = AsyncMock(return_value=ActionResult.success("tool_result"))
    return registry


@pytest.fixture
def executor(
    event_bus: EventBus, tools: MagicMock, websocket: AsyncMock
) -> ToolCallExecutor:
    return ToolCallExecutor(event_bus, tools, websocket)


def make_function_call_item(
    name: str = "get_weather",
    call_id: str = "call_001",
    response_id: str = "resp_001",
    arguments: dict | None = None,
) -> FunctionCallItem:
    return FunctionCallItem(
        event_id=f"evt_{call_id}",
        call_id=call_id,
        item_id=f"item_{call_id}",
        output_index=0,
        response_id=response_id,
        name=name,
        arguments=arguments or {},
    )


def make_response_done(response_id: str = "resp_001") -> ResponseDoneEvent:
    return ResponseDoneEvent(
        type=RealtimeServerEvent.RESPONSE_DONE,
        event_id=f"evt_done_{response_id}",
        response=RealtimeResponseObject(id=response_id),
    )


def make_tool(
    name: str = "get_weather",
    result_instruction: str | None = None,
    respond: bool = True,
    kind: ActionKind = ActionKind.GENERIC,
) -> MagicMock:
    tool = MagicMock(spec=Tool)
    tool.name = name
    tool.result_instruction = result_instruction
    tool.respond = respond
    tool.kind = kind
    return tool


class TestToolBatching:
    @pytest.mark.asyncio
    async def test_single_tool_waits_for_response_done(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()

        await event_bus.dispatch(make_function_call_item())
        assert websocket.send.call_count == 0

        await event_bus.dispatch(make_response_done())
        sent_types = [type(call.args[0]) for call in websocket.send.call_args_list]
        assert sent_types == [
            ConversationItemCreateEvent,
            ConversationResponseCreateEvent,
        ]

    @pytest.mark.asyncio
    async def test_parallel_tools_send_all_outputs_then_one_response(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.side_effect = make_tool
        tools.execute.side_effect = [
            ActionResult.success("sunny"),
            ActionResult.success("free"),
        ]

        await event_bus.dispatch(make_function_call_item("get_weather", "call_weather"))
        await event_bus.dispatch(
            make_function_call_item("get_calendar", "call_calendar")
        )
        await event_bus.dispatch(make_response_done())

        events = [call.args[0] for call in websocket.send.call_args_list]
        assert sum(isinstance(e, ConversationItemCreateEvent) for e in events) == 2
        assert sum(isinstance(e, ConversationResponseCreateEvent) for e in events) == 1
        assert isinstance(events[-1], ConversationResponseCreateEvent)
        assert [e.item.call_id for e in events[:-1]] == [
            "call_weather",
            "call_calendar",
        ]

    @pytest.mark.asyncio
    async def test_parallel_execution_starts_before_response_done(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        tools: MagicMock,
    ) -> None:
        started = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()

        async def execute(name: str, arguments: dict) -> ActionResult:
            started[0 if name == "first" else 1].set()
            await release.wait()
            return ActionResult.success(name)

        tools.get.side_effect = make_tool
        tools.execute.side_effect = execute
        await event_bus.dispatch(make_function_call_item("first", "call_1"))
        await event_bus.dispatch(make_function_call_item("second", "call_2"))
        await asyncio.gather(*(event.wait() for event in started))
        release.set()
        await event_bus.dispatch(make_response_done())

    @pytest.mark.asyncio
    async def test_failures_do_not_drop_other_outputs(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.side_effect = make_tool
        tools.execute.side_effect = [
            ActionResult.fail("boom"),
            ActionResult.success("ok"),
        ]

        await event_bus.dispatch(make_function_call_item("broken", "call_1"))
        await event_bus.dispatch(make_function_call_item("working", "call_2"))
        await event_bus.dispatch(make_response_done())

        events = [call.args[0] for call in websocket.send.call_args_list]
        assert len(events) == 3
        assert events[0].item.output == "boom"
        assert events[1].item.output == "ok"
        assert isinstance(events[2], ConversationResponseCreateEvent)

    @pytest.mark.asyncio
    async def test_result_instructions_are_merged(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.side_effect = [
            make_tool("first", "Explain the weather."),
            make_tool("second", "Mention the appointment."),
        ]

        await event_bus.dispatch(make_function_call_item("first", "call_1"))
        await event_bus.dispatch(make_function_call_item("second", "call_2"))
        await event_bus.dispatch(make_response_done())

        response = websocket.send.call_args_list[-1].args[0]
        assert response.response.instructions == (
            "Explain the weather.\nMention the appointment."
        )

    @pytest.mark.asyncio
    async def test_structured_result_is_passed_through_pre_serialized(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        class WeatherPayload(BaseModel):
            city: str
            temperature: int

        tools.get.return_value = make_tool()
        tools.execute.return_value = ActionResult.success(
            WeatherPayload(city="Berlin", temperature=18).model_dump_json()
        )

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(make_response_done())

        item = websocket.send.call_args_list[0].args[0]
        assert item.item.output == '{"city":"Berlin","temperature":18}'

    @pytest.mark.asyncio
    async def test_executes_with_arguments(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        await event_bus.dispatch(make_function_call_item(arguments={"city": "Berlin"}))
        await event_bus.dispatch(make_response_done())
        tools.execute.assert_awaited_once_with("get_weather", {"city": "Berlin"})

    @pytest.mark.asyncio
    async def test_silent_result_sends_output_without_creating_response(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        tools.execute.return_value = ActionResult.success("done", respond=False)

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(make_response_done())

        events = [call.args[0] for call in websocket.send.call_args_list]
        assert len(events) == 1
        assert isinstance(events[0], ConversationItemCreateEvent)
        assert events[0].item.output == "done"

    @pytest.mark.asyncio
    async def test_silent_batch_dispatches_tool_lifecycle(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        tools: MagicMock,
    ) -> None:
        started: list[ToolExecutionStartedEvent] = []
        completed: list[ToolExecutionCompletedEvent] = []
        executed: list[ToolExecutedEvent] = []
        event_bus.on(ToolExecutionStartedEvent, started.append)
        event_bus.on(ToolExecutionCompletedEvent, completed.append)
        event_bus.on(ToolExecutedEvent, executed.append)
        tools.get.return_value = make_tool(
            respond=False,
            kind=ActionKind.MUTATE,
        )

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(make_response_done())

        assert [event.response_id for event in started] == ["resp_001"]
        assert len(completed) == 1
        assert completed[0].response_id == "resp_001"
        assert completed[0].response_pending is False
        assert len(executed) == 1
        assert executed[0].name == "get_weather"
        assert executed[0].action_kind is ActionKind.MUTATE
        assert executed[0].silent is True
        assert executed[0].result == "tool_result"

    @pytest.mark.asyncio
    async def test_execution_event_uses_result_response_override(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        tools: MagicMock,
    ) -> None:
        executed: list[ToolExecutedEvent] = []
        event_bus.on(ToolExecutedEvent, executed.append)
        tools.get.return_value = make_tool(respond=False)
        tools.execute.return_value = ActionResult.success(respond=True)

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(make_response_done())

        assert executed[0].silent is False

    @pytest.mark.asyncio
    async def test_silent_action_sends_output_without_creating_response(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool(respond=False)

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(make_response_done())

        assert len(websocket.send.call_args_list) == 1

    @pytest.mark.asyncio
    async def test_silent_action_failure_creates_response(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool(respond=False)
        tools.execute.return_value = ActionResult.fail("boom")

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(make_response_done())

        events = [call.args[0] for call in websocket.send.call_args_list]
        assert isinstance(events[-1], ConversationResponseCreateEvent)

    @pytest.mark.asyncio
    async def test_mixed_batch_creates_response(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.side_effect = make_tool
        tools.execute.side_effect = [
            ActionResult.success("quiet", respond=False),
            ActionResult.success("loud", respond=True),
        ]

        await event_bus.dispatch(make_function_call_item("quiet", "call_1"))
        await event_bus.dispatch(make_function_call_item("loud", "call_2"))
        await event_bus.dispatch(make_response_done())

        events = [call.args[0] for call in websocket.send.call_args_list]
        assert isinstance(events[-1], ConversationResponseCreateEvent)

    @pytest.mark.asyncio
    async def test_silent_failure_override_suppresses_response(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        tools.execute.return_value = ActionResult.fail("boom", respond=False)

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(make_response_done())

        assert len(websocket.send.call_args_list) == 1


class TestUnknownTools:
    @pytest.mark.asyncio
    async def test_unknown_tool_is_ignored(
        self,
        event_bus: EventBus,
        executor: ToolCallExecutor,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        await event_bus.dispatch(make_function_call_item(name="missing"))
        await event_bus.dispatch(make_response_done())
        websocket.send.assert_not_called()
        tools.execute.assert_not_called()
