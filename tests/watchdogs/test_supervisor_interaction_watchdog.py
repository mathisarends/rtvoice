import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.agent.views import SupervisorClarificationNeeded, SupervisorDone
from rtvoice.events.bus import EventBus
from rtvoice.events.views import (
    CancelSupervisorCommand,
    SupervisorFinishedEvent,
    SupervisorStartedEvent,
    UpdateSessionToolsCommand,
    UpdateSupervisorCommand,
)
from rtvoice.handler import SupervisorCoordinator
from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
    FunctionCallItem,
    ToolChoiceMode,
)
from rtvoice.tools import ToolContext, Tools
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
def tools() -> Tools:
    return Tools()


@pytest.fixture
def watchdog(
    event_bus: EventBus, tools: Tools, websocket: AsyncMock
) -> SupervisorCoordinator:
    return SupervisorCoordinator(event_bus, tools, websocket, make_supervisor())


def make_function_call_item(
    name: str = "supervisor",
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
    tools: Tools,
    name: str = "supervisor",
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
    tools: Tools,
    name: str = "supervisor",
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
    async def _tool(
        query: str | None = None, clarification_answer: str | None = None
    ) -> str:
        kwargs = {"query": query} if query is not None else {}
        _ = clarification_answer
        calls.append(kwargs)
        return "tool_result"

    tool = tools.get(name)
    assert tool is not None
    return tool, calls


def make_supervisor() -> MagicMock:
    agent = MagicMock()
    agent.name = "supervisor"
    return agent


class TestNonSupervisorToolIgnored:
    @pytest.mark.asyncio
    async def test_unregistered_tool_name_does_not_send(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        websocket: AsyncMock,
        tools: Tools,
    ) -> None:
        register_tool(tools, name="other")
        await event_bus.dispatch(make_function_call_item(name="other"))

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_unregistered_tool_name_does_not_execute(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        _, calls = register_tool_with_calls(tools, name="other")
        await event_bus.dispatch(make_function_call_item(name="other"))

        assert calls == []


class TestToolCallHandling:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SupervisorCoordinator) -> None:
        pass

    @pytest.mark.asyncio
    async def test_tool_not_found_does_not_send(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        websocket: AsyncMock,
        tools: Tools,
    ) -> None:
        await event_bus.dispatch(make_function_call_item())

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_not_found_does_not_execute(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        await event_bus.dispatch(make_function_call_item())
        tool = tools.get("supervisor")
        assert tool is None

    @pytest.mark.asyncio
    async def test_sends_holding_response_immediately_when_configured(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        websocket: AsyncMock,
        tools: Tools,
    ) -> None:
        register_tool(tools, holding_instruction="Please wait briefly.")

        await event_bus.dispatch(make_function_call_item())

        assert websocket.send.call_count >= 1
        sent = websocket.send.call_args_list[0][0][0]
        assert isinstance(sent, ConversationResponseCreateEvent)

    @pytest.mark.asyncio
    async def test_does_not_send_holding_response_without_instruction(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        websocket: AsyncMock,
        tools: Tools,
    ) -> None:
        register_tool(tools)

        await event_bus.dispatch(make_function_call_item())
        await asyncio.sleep(0.05)

        sent_payloads = [c.args[0] for c in websocket.send.call_args_list]
        response_events = [
            event
            for event in sent_payloads
            if isinstance(event, ConversationResponseCreateEvent)
        ]

        assert isinstance(sent_payloads[0], ConversationItemCreateEvent)
        assert len(response_events) == 1

    @pytest.mark.asyncio
    async def test_dispatches_started_event(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        received: list[SupervisorStartedEvent] = []

        async def capture(e: SupervisorStartedEvent) -> None:
            received.append(e)

        event_bus.subscribe(SupervisorStartedEvent, capture)

        await event_bus.dispatch(make_function_call_item())

        assert len(received) == 1
        assert isinstance(received[0], SupervisorStartedEvent)

    @pytest.mark.asyncio
    async def test_executes_tool_with_arguments(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
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
        watchdog: SupervisorCoordinator,
        websocket: AsyncMock,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        _block = asyncio.Event()

        async def blocking_execute(query: str | None = None) -> str:
            await _block.wait()
            return "done"

        tool = tools.get("supervisor")
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
    def setup_supervisor(self, watchdog: SupervisorCoordinator) -> None:
        pass

    @pytest.mark.asyncio
    async def test_delivers_function_call_output_after_holding(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        websocket: AsyncMock,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        tool = tools.get("supervisor")
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
        watchdog: SupervisorCoordinator,
        websocket: AsyncMock,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        tool = tools.get("supervisor")
        assert tool is not None

        async def done_tool(query: str | None = None) -> str:
            return "job_done"

        tool.function = done_tool

        await event_bus.dispatch(make_function_call_item(call_id="call_lr"))
        await asyncio.sleep(0.05)

        sent_types = [type(c.args[0]) for c in websocket.send.call_args_list]
        assert ConversationResponseCreateEvent in sent_types

    @pytest.mark.asyncio
    async def test_sends_response_create_when_supervisor_result_returned(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        websocket: AsyncMock,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        tool = tools.get("supervisor")
        assert tool is not None

        async def silent_done_tool(query: str | None = None) -> SupervisorDone:
            return SupervisorDone(message="job_done")

        tool.function = silent_done_tool

        await event_bus.dispatch(make_function_call_item(call_id="call_silent"))
        await asyncio.sleep(0.05)

        sent_payloads = [c.args[0] for c in websocket.send.call_args_list]
        response_events = [
            event
            for event in sent_payloads
            if isinstance(event, ConversationResponseCreateEvent)
        ]
        item_events = [
            event
            for event in sent_payloads
            if isinstance(event, ConversationItemCreateEvent)
        ]

        assert len(response_events) == 1
        assert len(item_events) == 1

    @pytest.mark.asyncio
    async def test_pending_cleared_after_result_delivered(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        tool = tools.get("supervisor")
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
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        tool = tools.get("supervisor")
        assert tool is not None

        async def done_tool(query: str | None = None) -> str:
            return "done"

        tool.function = done_tool
        received: list[SupervisorFinishedEvent] = []

        async def capture(e: SupervisorFinishedEvent) -> None:
            received.append(e)

        event_bus.subscribe(SupervisorFinishedEvent, capture)

        await event_bus.dispatch(make_function_call_item())
        await asyncio.sleep(0.05)

        assert len(received) == 1
        assert isinstance(received[0], SupervisorFinishedEvent)

    @pytest.mark.asyncio
    async def test_result_not_delivered_before_tool_completes(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        websocket: AsyncMock,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        block = asyncio.Event()

        async def blocking_execute(query: str | None = None) -> str:
            await block.wait()
            return "done"

        tool = tools.get("supervisor")
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
    def setup_supervisor(self, watchdog: SupervisorCoordinator) -> None:
        pass

    @pytest.mark.asyncio
    async def test_cancel_clears_pending(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        _block = asyncio.Event()

        async def blocking_execute(query: str | None = None) -> str:
            await _block.wait()
            return "done"

        tool = tools.get("supervisor")
        assert tool is not None
        tool.function = blocking_execute

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(CancelSupervisorCommand())

        assert watchdog._active is None

    @pytest.mark.asyncio
    async def test_cancel_dispatches_finished_event(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        _block = asyncio.Event()

        async def blocking_execute(query: str | None = None) -> str:
            await _block.wait()
            return "done"

        tool = tools.get("supervisor")
        assert tool is not None
        tool.function = blocking_execute

        received: list[SupervisorFinishedEvent] = []

        async def capture(e: SupervisorFinishedEvent) -> None:
            received.append(e)

        event_bus.subscribe(SupervisorFinishedEvent, capture)

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(CancelSupervisorCommand())

        assert len(received) == 1
        assert isinstance(received[0], SupervisorFinishedEvent)

    @pytest.mark.asyncio
    async def test_cancel_without_pending_is_safe(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        websocket: AsyncMock,
    ) -> None:
        await event_bus.dispatch(CancelSupervisorCommand())

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_cancels_result_task(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        _block = asyncio.Event()

        async def blocking_execute(query: str | None = None) -> str:
            await _block.wait()
            return "done"

        tool = tools.get("supervisor")
        assert tool is not None
        tool.function = blocking_execute

        await event_bus.dispatch(make_function_call_item())
        result_task = watchdog._active.execution_task

        await event_bus.dispatch(CancelSupervisorCommand())
        await asyncio.sleep(0.01)

        assert result_task.cancelled()


class TestCancelTool:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SupervisorCoordinator) -> None:
        pass

    @pytest.mark.asyncio
    async def test_cancel_tool_ejected_after_result_delivered(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        tool = tools.get("supervisor")
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

        assert tools.get("cancel_supervisor") is None
        assert tools.get("update_supervisor") is None
        assert len(received) >= 1


class TestUpdateSupervisor:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SupervisorCoordinator) -> None:
        pass

    @pytest.mark.asyncio
    async def test_update_command_pushes_message_to_active_supervisor(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        supervisor = watchdog._supervisor
        supervisor.update = AsyncMock()
        register_tool(tools)
        block = asyncio.Event()

        async def blocking_execute(query: str | None = None) -> str:
            await block.wait()
            return "done"

        tool = tools.get("supervisor")
        assert tool is not None
        tool.function = blocking_execute

        await event_bus.dispatch(make_function_call_item())
        await event_bus.dispatch(
            UpdateSupervisorCommand(message="Focus on the European market")
        )

        supervisor.update.assert_awaited_once_with("Focus on the European market")
        block.set()
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_update_command_without_active_supervisor_is_ignored(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
    ) -> None:
        supervisor = watchdog._supervisor
        supervisor.update = AsyncMock()

        await event_bus.dispatch(UpdateSupervisorCommand(message="Use EU context"))

        supervisor.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_tool_dispatches_update_command(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        supervisor = watchdog._supervisor
        supervisor.update = AsyncMock()
        tools.set_context(ToolContext(event_bus=event_bus))
        register_tool(tools)
        block = asyncio.Event()

        async def blocking_execute(query: str | None = None) -> str:
            await block.wait()
            return "done"

        tool = tools.get("supervisor")
        assert tool is not None
        tool.function = blocking_execute

        await event_bus.dispatch(make_function_call_item())
        result = await tools.execute(
            "update_supervisor", {"message": "Prioritize Europe"}
        )

        assert result == "The supervisor has received the update."
        supervisor.update.assert_awaited_once_with("Prioritize Europe")
        block.set()
        await asyncio.sleep(0.05)


class TestClarificationFlow:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SupervisorCoordinator) -> None:
        pass

    @pytest.mark.asyncio
    async def test_clarification_result_sets_awaiting_flag_and_sends_prompt(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        websocket: AsyncMock,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        tool = tools.get("supervisor")
        assert tool is not None

        async def clarify_tool(
            query: str | None = None,
        ) -> SupervisorClarificationNeeded:
            return SupervisorClarificationNeeded(
                question="Which date should I use?",
                resume_history=[],
                clarify_call_id="clarify_1",
            )

        tool.function = clarify_tool

        await event_bus.dispatch(make_function_call_item(call_id="call_clarify"))
        await asyncio.sleep(0.05)

        assert watchdog._awaiting_clarification is True

        sent_payloads = [c.args[0] for c in websocket.send.call_args_list]
        output_events = [
            event
            for event in sent_payloads
            if isinstance(event, ConversationItemCreateEvent)
        ]
        response_events = [
            event
            for event in sent_payloads
            if isinstance(event, ConversationResponseCreateEvent)
        ]

        assert any(
            "Clarification needed" in event.item.output for event in output_events
        )
        assert any(
            event.response is not None
            and "Which date should I use?" in event.response.instructions
            and event.response.tool_choice == ToolChoiceMode.AUTO
            for event in response_events
        )

    @pytest.mark.asyncio
    async def test_non_supervisor_tool_call_is_ignored_while_waiting_for_clarification(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        websocket: AsyncMock,
        tools: Tools,
    ) -> None:
        register_tool(tools, name="supervisor")
        register_tool(tools, name="other_job")

        watchdog._awaiting_clarification = True

        await event_bus.dispatch(
            make_function_call_item(name="other_job", call_id="call_ignored")
        )

        websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_supervisor_call_with_clarification_answer_is_allowed(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        _, calls = register_tool_with_calls(tools, name="supervisor")
        watchdog._awaiting_clarification = True

        await event_bus.dispatch(
            make_function_call_item(
                call_id="call_resume",
                arguments={"query": "resume answer", "clarification_answer": "Tuesday"},
            )
        )
        await asyncio.sleep(0.05)

        assert calls == [{"query": "resume answer"}]
        assert watchdog._awaiting_clarification is False


class TestSessionToolsSyncing:
    @pytest.fixture(autouse=True)
    def setup_supervisor(self, watchdog: SupervisorCoordinator) -> None:
        pass

    @pytest.mark.asyncio
    async def test_successful_supervisor_run_syncs_tools_on_inject_and_eject(
        self,
        event_bus: EventBus,
        watchdog: SupervisorCoordinator,
        tools: Tools,
    ) -> None:
        register_tool(tools)
        received: list[UpdateSessionToolsCommand] = []

        async def capture(event: UpdateSessionToolsCommand) -> None:
            received.append(event)

        event_bus.subscribe(UpdateSessionToolsCommand, capture)

        await event_bus.dispatch(make_function_call_item(call_id="call_sync"))
        await asyncio.sleep(0.05)

        assert len(received) == 2
        assert any(tool.name == "cancel_supervisor" for tool in received[0].tools)
        assert any(tool.name == "update_supervisor" for tool in received[0].tools)
        assert all(tool.name != "cancel_supervisor" for tool in received[1].tools)
        assert all(tool.name != "update_supervisor" for tool in received[1].tools)
