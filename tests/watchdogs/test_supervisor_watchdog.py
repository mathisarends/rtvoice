import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.events.views import (
    CancelSubAgentCommand,
    SubAgentFinishedEvent,
    SubAgentStartedEvent,
    UpdateSessionToolsCommand,
)
from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
    FunctionCallItem,
)
from rtvoice.tools import RealtimeTools
from rtvoice.tools.registry.views import Tool
from rtvoice.watchdogs.subagent.subagent_interaction import SubAgentInteractionWatchdog


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def websocket() -> AsyncMock:
    ws = AsyncMock()
    ws.send = AsyncMock()
    return ws


@pytest.fixture
def tools() -> RealtimeTools:
    return RealtimeTools()


@pytest.fixture
def watchdog(
    event_bus: EventBus, tools: RealtimeTools, websocket: AsyncMock
) -> SubAgentInteractionWatchdog:
    return SubAgentInteractionWatchdog(event_bus, tools, websocket)


def make_function_call_item(
    name: str = "slow_job",
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


def register_tool(
    tools: RealtimeTools,
    name: str = "slow_job",
    result_instruction: str | None = None,
    holding_instruction: str | None = None,
) -> Tool:
    @tools.action(
        "Test tool",
        name=name,
        result_instruction=result_instruction,
        holding_instruction=holding_instruction,
    )
    async def _tool(query: str | None = None) -> str:
        return "tool_result"

    tool = tools.get(name)
    assert tool is not None
    return tool


def register_tool_with_calls(
    tools: RealtimeTools,
    name: str = "slow_job",
    result_instruction: str | None = None,
    holding_instruction: str | None = None,
) -> tuple[Tool, list[dict]]:
    calls: list[dict] = []

    @tools.action(
        "Test tool",
        name=name,
        result_instruction=result_instruction,
        holding_instruction=holding_instruction,
    )
    async def _tool(query: str | None = None) -> str:
        kwargs = {"query": query} if query is not None else {}
        calls.append(kwargs)
        return "tool_result"

    tool = tools.get(name)
    assert tool is not None
    return tool, calls


def make_subagent() -> MagicMock:
    agent = MagicMock()
    agent.name = "supervisor"
    agent._attach_channel.side_effect = lambda channel: channel.close()
    return agent


class TestNonSupervisorToolIgnored:
    @pytest.mark.asyncio
    async def test_unregistered_tool_name_does_not_send(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        websocket: AsyncMock,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools, name="other")
        watchdog.register_subagent("slow_job", make_subagent())

        await event_bus.dispatch(make_function_call_item(name="other"))

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_unregistered_tool_name_does_not_execute(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        tools: RealtimeTools,
    ) -> None:
        _, calls = register_tool_with_calls(tools, name="other")
        watchdog.register_subagent("slow_job", make_subagent())

        await event_bus.dispatch(make_function_call_item(name="other"))

        assert calls == []


class TestToolCallHandling:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SubAgentInteractionWatchdog) -> None:
        watchdog.register_subagent("slow_job", make_subagent())

    @pytest.mark.asyncio
    async def test_tool_not_found_does_not_send(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        websocket: AsyncMock,
        tools: RealtimeTools,
    ) -> None:
        await event_bus.dispatch(make_function_call_item())

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_not_found_does_not_execute(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        tools: RealtimeTools,
    ) -> None:
        await event_bus.dispatch(make_function_call_item())
        tool = tools.get("slow_job")
        assert tool is None

    @pytest.mark.asyncio
    async def test_sends_holding_response_immediately(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        websocket: AsyncMock,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools)

        await event_bus.dispatch(make_function_call_item())

        websocket.send.assert_called_once()
        sent = websocket.send.call_args[0][0]
        assert isinstance(sent, ConversationResponseCreateEvent)

    @pytest.mark.asyncio
    async def test_dispatches_started_event(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools)
        received: list[SubAgentStartedEvent] = []

        async def capture(e: SubAgentStartedEvent) -> None:
            received.append(e)

        event_bus.subscribe(SubAgentStartedEvent, capture)

        await event_bus.dispatch(make_function_call_item())

        assert len(received) == 1
        assert received[0].agent_name == "slow_job"

    @pytest.mark.asyncio
    async def test_executes_tool_with_arguments(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        tools: RealtimeTools,
    ) -> None:
        _, calls = register_tool_with_calls(tools)

        await event_bus.dispatch(
            make_function_call_item(arguments={"query": "Berlin weather"})
        )

        assert calls == [{"query": "Berlin weather"}]

    @pytest.mark.asyncio
    async def test_duplicate_call_sends_already_in_progress(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        websocket: AsyncMock,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools)
        _block = asyncio.Event()

        async def blocking_execute(query: str | None = None) -> str:
            await _block.wait()
            return "done"

        tool = tools.get("slow_job")
        assert tool is not None
        tool.function = blocking_execute

        await event_bus.dispatch(make_function_call_item(call_id="call_1"))
        await event_bus.dispatch(make_function_call_item(call_id="call_2"))

        all_sent = [c.args[0] for c in websocket.send.call_args_list]
        fn_outputs = [s for s in all_sent if isinstance(s, ConversationItemCreateEvent)]
        assert any(
            "already in progress" in s.item.output.lower()
            for s in fn_outputs
            if hasattr(s.item, "output")
        )


class TestResultDelivery:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SubAgentInteractionWatchdog) -> None:
        watchdog.register_subagent("slow_job", make_subagent())

    @pytest.mark.asyncio
    async def test_delivers_function_call_output_after_holding(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        websocket: AsyncMock,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools)
        tool = tools.get("slow_job")
        assert tool is not None

        async def done_tool(query: str | None = None) -> str:
            return "job_done"

        tool.function = done_tool

        await event_bus.dispatch(make_function_call_item(call_id="call_lr"))
        await asyncio.sleep(0.05)

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ConversationItemCreateEvent in sent_types

    @pytest.mark.asyncio
    async def test_sends_response_create_after_result(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        websocket: AsyncMock,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools)
        tool = tools.get("slow_job")
        assert tool is not None

        async def done_tool(query: str | None = None) -> str:
            return "job_done"

        tool.function = done_tool

        await event_bus.dispatch(make_function_call_item(call_id="call_lr"))
        await asyncio.sleep(0.05)

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ConversationResponseCreateEvent in sent_types

    @pytest.mark.asyncio
    async def test_pending_cleared_after_result_delivered(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools)
        tool = tools.get("slow_job")
        assert tool is not None

        async def done_tool(query: str | None = None) -> str:
            return "done"

        tool.function = done_tool

        await event_bus.dispatch(make_function_call_item())
        await asyncio.sleep(0.05)

        assert watchdog._active is None

    @pytest.mark.asyncio
    async def test_dispatches_finished_event_after_result(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools)
        tool = tools.get("slow_job")
        assert tool is not None

        async def done_tool(query: str | None = None) -> str:
            return "done"

        tool.function = done_tool
        received: list[SubAgentFinishedEvent] = []

        async def capture(e: SubAgentFinishedEvent) -> None:
            received.append(e)

        event_bus.subscribe(SubAgentFinishedEvent, capture)

        await event_bus.dispatch(make_function_call_item())
        await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0].agent_name == "slow_job"

    @pytest.mark.asyncio
    async def test_result_not_delivered_before_tool_completes(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        websocket: AsyncMock,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools)
        block = asyncio.Event()

        async def blocking_execute(query: str | None = None) -> str:
            await block.wait()
            return "done"

        tool = tools.get("slow_job")
        assert tool is not None
        tool.function = blocking_execute

        await event_bus.dispatch(make_function_call_item())
        await asyncio.sleep(0.05)

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ConversationItemCreateEvent not in sent_types

        block.set()
        await asyncio.sleep(0.05)

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ConversationItemCreateEvent in sent_types


class TestCancelSupervisor:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SubAgentInteractionWatchdog) -> None:
        watchdog.register_subagent("slow_job", make_subagent())

    @pytest.mark.asyncio
    async def test_cancel_clears_pending(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools)
        _block = asyncio.Event()

        async def blocking_execute(query: str | None = None) -> str:
            await _block.wait()
            return "done"

        tool = tools.get("slow_job")
        assert tool is not None
        tool.function = blocking_execute

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(CancelSubAgentCommand())

        assert watchdog._active is None

    @pytest.mark.asyncio
    async def test_cancel_dispatches_finished_event(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools)
        _block = asyncio.Event()

        async def blocking_execute(query: str | None = None) -> str:
            await _block.wait()
            return "done"

        tool = tools.get("slow_job")
        assert tool is not None
        tool.function = blocking_execute

        received: list[SubAgentFinishedEvent] = []

        async def capture(e: SubAgentFinishedEvent) -> None:
            received.append(e)

        event_bus.subscribe(SubAgentFinishedEvent, capture)

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(CancelSubAgentCommand())

        assert len(received) == 1
        assert received[0].agent_name == "slow_job"

    @pytest.mark.asyncio
    async def test_cancel_without_pending_is_safe(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        websocket: AsyncMock,
    ) -> None:
        await event_bus.dispatch(CancelSubAgentCommand())

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_cancels_result_task(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools)
        _block = asyncio.Event()

        async def blocking_execute(query: str | None = None) -> str:
            await _block.wait()
            return "done"

        tool = tools.get("slow_job")
        assert tool is not None
        tool.function = blocking_execute

        await event_bus.dispatch(make_function_call_item())
        result_task = watchdog._active.execution_task

        await event_bus.dispatch(CancelSubAgentCommand())
        await asyncio.sleep(0.01)

        assert result_task.cancelled()


class TestCancelTool:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SubAgentInteractionWatchdog) -> None:
        watchdog.register_subagent("slow_job", make_subagent())

    @pytest.mark.asyncio
    async def test_cancel_tool_ejected_after_result_delivered(
        self,
        event_bus: EventBus,
        watchdog: SubAgentInteractionWatchdog,
        tools: RealtimeTools,
    ) -> None:
        register_tool(tools)
        tool = tools.get("slow_job")
        assert tool is not None

        async def done_tool(query: str | None = None) -> str:
            return "done"

        tool.function = done_tool

        received: list[UpdateSessionToolsCommand] = []

        async def capture(e: UpdateSessionToolsCommand) -> None:
            received.append(e)

        event_bus.subscribe(UpdateSessionToolsCommand, capture)

        await event_bus.dispatch(make_function_call_item())
        await asyncio.sleep(0.05)

        assert tools.get("cancel_agent") is None
        assert len(received) >= 1
