import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.events.views import (
    CancelSupervisorCommand,
    SupervisorFinishedEvent,
    SupervisorStartedEvent,
    UpdateSessionToolsCommand,
    UserTranscriptCompletedEvent,
)
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
from rtvoice.watchdogs.supervisor.watchdog import SupervisorWatchdog


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
    t._registry = MagicMock()
    t._registry.tools = {}
    t.get_tool_schema = MagicMock(return_value=[])
    return t


@pytest.fixture
def watchdog(
    event_bus: EventBus, tools: MagicMock, websocket: AsyncMock
) -> SupervisorWatchdog:
    return SupervisorWatchdog(event_bus, tools, websocket)


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


def make_tool(name: str = "slow_job") -> MagicMock:
    tool = MagicMock(spec=Tool)
    tool.name = name
    tool.result_instruction = None
    tool.holding_instruction = None
    return tool


def make_cancel_tool(name: str = "cancel_job") -> MagicMock:
    tool = MagicMock(spec=Tool)
    tool.name = name
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


async def complete_holding_phase(
    event_bus: EventBus, response_id: str = "resp_hold"
) -> None:
    await event_bus.dispatch(make_response_created(response_id))
    await event_bus.dispatch(make_response_done(response_id))


class TestNonSupervisorToolIgnored:
    @pytest.mark.asyncio
    async def test_unregistered_tool_name_does_not_send(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool(name="other")
        watchdog.register_supervisor("slow_job", make_supervisor_agent())

        await event_bus.dispatch(make_function_call_item(name="other"))

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_unregistered_tool_name_does_not_execute(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool(name="other")
        watchdog.register_supervisor("slow_job", make_supervisor_agent())

        await event_bus.dispatch(make_function_call_item(name="other"))

        tools.execute.assert_not_called()


class TestToolCallHandling:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SupervisorWatchdog) -> None:
        watchdog.register_supervisor("slow_job", make_supervisor_agent())

    @pytest.mark.asyncio
    async def test_tool_not_found_does_not_send(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = None

        await event_bus.dispatch(make_function_call_item())

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_not_found_does_not_execute(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = None

        await event_bus.dispatch(make_function_call_item())

        tools.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_holding_response_immediately(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()

        await event_bus.dispatch(make_function_call_item())

        websocket.send.assert_called_once()
        sent = websocket.send.call_args[0][0]
        assert isinstance(sent, ConversationResponseCreateEvent)

    @pytest.mark.asyncio
    async def test_dispatches_started_event(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        received: list[SupervisorStartedEvent] = []

        async def capture(e: SupervisorStartedEvent) -> None:
            received.append(e)

        event_bus.subscribe(SupervisorStartedEvent, capture)

        await event_bus.dispatch(make_function_call_item())

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_executes_tool_with_arguments(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()

        await event_bus.dispatch(
            make_function_call_item(arguments={"query": "Berlin weather"})
        )

        tools.execute.assert_called_once_with("slow_job", {"query": "Berlin weather"})

    @pytest.mark.asyncio
    async def test_duplicate_call_sends_already_in_progress(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        _block = asyncio.Event()

        async def blocking_execute(name: str, arguments: dict) -> str:
            await _block.wait()
            return "done"

        tools.execute = blocking_execute

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
    def setup_supervisor(self, watchdog: SupervisorWatchdog) -> None:
        watchdog.register_supervisor("slow_job", make_supervisor_agent())

    @pytest.mark.asyncio
    async def test_delivers_function_call_output_after_holding(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        tools.execute = AsyncMock(return_value="job_done")

        await event_bus.dispatch(make_function_call_item(call_id="call_lr"))
        await complete_holding_phase(event_bus)
        await asyncio.sleep(0.05)

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ConversationItemCreateEvent in sent_types

    @pytest.mark.asyncio
    async def test_sends_response_create_after_result(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        tools.execute = AsyncMock(return_value="job_done")

        await event_bus.dispatch(make_function_call_item(call_id="call_lr"))
        await complete_holding_phase(event_bus)
        await asyncio.sleep(0.05)

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ConversationResponseCreateEvent in sent_types

    @pytest.mark.asyncio
    async def test_pending_cleared_after_result_delivered(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        tools.execute = AsyncMock(return_value="done")

        await event_bus.dispatch(make_function_call_item())
        await complete_holding_phase(event_bus)
        await asyncio.sleep(0.05)

        assert watchdog._pending is None

    @pytest.mark.asyncio
    async def test_dispatches_finished_event_after_result(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        tools.execute = AsyncMock(return_value="done")
        received: list[SupervisorFinishedEvent] = []

        async def capture(e: SupervisorFinishedEvent) -> None:
            received.append(e)

        event_bus.subscribe(SupervisorFinishedEvent, capture)

        await event_bus.dispatch(make_function_call_item())
        await complete_holding_phase(event_bus)
        await asyncio.sleep(0.05)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_result_not_delivered_before_holding_done(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        websocket: AsyncMock,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        tools.execute = AsyncMock(return_value="done")

        await event_bus.dispatch(make_function_call_item())
        await asyncio.sleep(0.05)

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ConversationItemCreateEvent not in sent_types


class TestResponseTracking:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SupervisorWatchdog) -> None:
        watchdog.register_supervisor("slow_job", make_supervisor_agent())

    @pytest.mark.asyncio
    async def test_first_response_created_sets_holding_id(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        _block = asyncio.Event()

        async def blocking_execute(name: str, arguments: dict) -> str:
            await _block.wait()
            return "done"

        tools.execute = blocking_execute

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(make_response_created("resp_hold"))

        assert watchdog._pending.holding_response_id == "resp_hold"

    @pytest.mark.asyncio
    async def test_response_done_for_wrong_id_does_not_set_holding_done(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        _block = asyncio.Event()

        async def blocking_execute(name: str, arguments: dict) -> str:
            await _block.wait()
            return "done"

        tools.execute = blocking_execute

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(make_response_created("resp_hold"))
        await event_bus.dispatch(make_response_done("resp_other"))

        assert not watchdog._pending.holding_done.is_set()

    @pytest.mark.asyncio
    async def test_response_done_with_correct_id_sets_holding_done(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        _block = asyncio.Event()

        async def blocking_execute(name: str, arguments: dict) -> str:
            await _block.wait()
            return "done"

        tools.execute = blocking_execute

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(make_response_created("resp_hold"))
        await event_bus.dispatch(make_response_done("resp_hold"))

        assert watchdog._pending.holding_done.is_set()


class TestClarificationResponse:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SupervisorWatchdog) -> None:
        watchdog.register_supervisor("slow_job", make_supervisor_agent())

    @pytest.mark.asyncio
    async def test_clarification_answer_resolves_future(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        _block = asyncio.Event()

        async def blocking_execute(name: str, arguments: dict) -> str:
            await _block.wait()
            return "done"

        tools.execute = blocking_execute

        await event_bus.dispatch(make_function_call_item())

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        watchdog._pending.supervisor_run.pending_clarification_future = future

        await event_bus.dispatch(
            UserTranscriptCompletedEvent(transcript="Paris", item_id="item_x")
        )

        assert future.done()
        assert future.result() == "Paris"

    @pytest.mark.asyncio
    async def test_clarification_without_pending_future_is_safe(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        _block = asyncio.Event()

        async def blocking_execute(name: str, arguments: dict) -> str:
            await _block.wait()
            return "done"

        tools.execute = blocking_execute

        await event_bus.dispatch(make_function_call_item())

        await event_bus.dispatch(
            UserTranscriptCompletedEvent(transcript="anything", item_id="item_x")
        )


class TestCancelSupervisor:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SupervisorWatchdog) -> None:
        watchdog.register_supervisor("slow_job", make_supervisor_agent())

    @pytest.mark.asyncio
    async def test_cancel_clears_pending(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        _block = asyncio.Event()

        async def blocking_execute(name: str, arguments: dict) -> str:
            await _block.wait()
            return "done"

        tools.execute = blocking_execute

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(CancelSupervisorCommand())

        assert watchdog._pending is None

    @pytest.mark.asyncio
    async def test_cancel_dispatches_finished_event(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        _block = asyncio.Event()

        async def blocking_execute(name: str, arguments: dict) -> str:
            await _block.wait()
            return "done"

        tools.execute = blocking_execute

        received: list[SupervisorFinishedEvent] = []

        async def capture(e: SupervisorFinishedEvent) -> None:
            received.append(e)

        event_bus.subscribe(SupervisorFinishedEvent, capture)

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(CancelSupervisorCommand())

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_cancel_without_pending_is_safe(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        websocket: AsyncMock,
    ) -> None:
        await event_bus.dispatch(CancelSupervisorCommand())

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_cancels_result_task(
        self,
        event_bus: EventBus,
        watchdog: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        _block = asyncio.Event()

        async def blocking_execute(name: str, arguments: dict) -> str:
            await _block.wait()
            return "done"

        tools.execute = blocking_execute

        await event_bus.dispatch(make_function_call_item())
        result_task = watchdog._pending.result_task

        await event_bus.dispatch(CancelSupervisorCommand())
        await asyncio.sleep(0.01)

        assert result_task.cancelled()


class TestCancelTool:
    @pytest.fixture
    def cancel_tool(self) -> MagicMock:
        return make_cancel_tool()

    @pytest.fixture
    def watchdog_with_cancel(
        self,
        event_bus: EventBus,
        tools: MagicMock,
        websocket: AsyncMock,
        cancel_tool: MagicMock,
    ) -> SupervisorWatchdog:
        wd = SupervisorWatchdog(event_bus, tools, websocket, cancel_tool=cancel_tool)
        wd.register_supervisor("slow_job", make_supervisor_agent())
        return wd

    @pytest.mark.asyncio
    async def test_cancel_tool_injected_when_supervisor_starts(
        self,
        event_bus: EventBus,
        watchdog_with_cancel: SupervisorWatchdog,
        tools: MagicMock,
        cancel_tool: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()

        await event_bus.dispatch(make_function_call_item())

        tools.inject_tool.assert_called_once_with(cancel_tool)

    @pytest.mark.asyncio
    async def test_cancel_tool_ejected_after_result_delivered(
        self,
        event_bus: EventBus,
        watchdog_with_cancel: SupervisorWatchdog,
        tools: MagicMock,
        cancel_tool: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        tools.execute = AsyncMock(return_value="done")

        await event_bus.dispatch(make_function_call_item())
        await complete_holding_phase(event_bus)
        await asyncio.sleep(0.05)

        tools.eject_tool.assert_called_with(cancel_tool.name)

    @pytest.mark.asyncio
    async def test_tools_update_dispatched_on_injection(
        self,
        event_bus: EventBus,
        watchdog_with_cancel: SupervisorWatchdog,
        tools: MagicMock,
    ) -> None:
        tools.get.return_value = make_tool()
        received: list[UpdateSessionToolsCommand] = []

        async def capture(e: UpdateSessionToolsCommand) -> None:
            received.append(e)

        event_bus.subscribe(UpdateSessionToolsCommand, capture)

        await event_bus.dispatch(make_function_call_item())

        assert len(received) >= 1
