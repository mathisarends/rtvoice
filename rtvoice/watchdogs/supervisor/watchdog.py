import asyncio
import json
import logging

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AssistantStoppedRespondingEvent,
    CancelSupervisorCommand,
    SupervisorFinishedEvent,
    SupervisorStartedEvent,
    UpdateSessionToolsCommand,
    UserTranscriptCompletedEvent,
)
from rtvoice.realtime.schemas import (
    ConversationResponseCreateEvent,
    FunctionCallItem,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ToolChoiceMode,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.supervisor import SupervisorAgent
from rtvoice.supervisor.channel import SupervisorChannel
from rtvoice.tools import Tools
from rtvoice.tools.registry.views import Tool
from rtvoice.watchdogs.supervisor.channel_relay import ChannelRelay
from rtvoice.watchdogs.supervisor.views import PendingSupervisorRun, PendingToolCall
from rtvoice.watchdogs.tool_calling.helpers import ToolCallWebSocketHelper

logger = logging.getLogger(__name__)

_DEFAULT_HOLDING_INSTRUCTION = (
    "The user's request is being processed. "
    "Say ONE brief sentence acknowledging it naturally "
    "(e.g. 'Let me look that up for you!'). "
    "Then stop. Do not keep talking and do not reveal any results."
)


class SupervisorWatchdog:
    def __init__(
        self,
        event_bus: EventBus,
        tools: Tools,
        websocket: RealtimeWebSocket,
        cancel_tool: Tool | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._tools = tools
        self._ws = ToolCallWebSocketHelper(websocket)
        self._websocket = websocket
        self._cancel_tool = cancel_tool
        self._pending: PendingToolCall | None = None
        self._supervisor_agent: SupervisorAgent | None = None
        self._supervisor_tool_name: str | None = None
        self._channel_relay = ChannelRelay(websocket)

        self._event_bus.subscribe(FunctionCallItem, self._handle_tool_call)
        self._event_bus.subscribe(ResponseCreatedEvent, self._on_response_created)
        self._event_bus.subscribe(ResponseDoneEvent, self._on_response_done)
        self._event_bus.subscribe(
            UserTranscriptCompletedEvent, self._on_clarification_response
        )
        self._event_bus.subscribe(CancelSupervisorCommand, self._on_cancel_supervisor)

        self._event_bus.subscribe(
            AssistantStoppedRespondingEvent, self._on_assistant_stopped
        )

    def register_supervisor(self, tool_name: str, agent: SupervisorAgent) -> None:
        self._supervisor_tool_name = tool_name
        self._supervisor_agent = agent
        logger.debug(
            "Supervisor agent '%s' registered on tool '%s'", agent.name, tool_name
        )

    async def _handle_tool_call(self, event: FunctionCallItem) -> None:
        if event.name != self._supervisor_tool_name:
            return

        tool = self._tools.get(event.name)
        if not tool:
            logger.error("Tool '%s' not found", event.name)
            return

        if self._pending is not None:
            logger.warning(
                "Duplicate supervisor call for '%s' — suppressing", event.name
            )
            await self._ws.send_function_call_output(
                event.call_id, "Request already in progress."
            )
            return

        logger.info(
            "Supervisor tool call: '%s' [args=%s]",
            event.name,
            json.dumps(event.arguments or {}, ensure_ascii=False),
        )

        channel = await self._create_supervisor_channel()
        result_task = asyncio.create_task(
            self._tools.execute(event.name, event.arguments or {})
        )

        pending = PendingToolCall(
            call_id=event.call_id,
            tool_name=event.name,
            result_task=result_task,
            tool=tool,
            channel=channel,
            supervisor_run=PendingSupervisorRun(),
        )
        self._pending = pending

        await self._event_bus.dispatch(SupervisorStartedEvent())
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                tool.holding_instruction or _DEFAULT_HOLDING_INSTRUCTION,
                tool_choice=ToolChoiceMode.NONE,
            )
        )
        asyncio.create_task(self._deliver_result(pending))

    async def _on_response_created(self, event: ResponseCreatedEvent) -> None:
        if self._pending is None:
            return
        pending = self._pending
        if pending.supervisor_run.response_id is None:
            pending.supervisor_run.response_id = event.response_id
            logger.debug(
                "Response '%s' tracked for '%s'",
                event.response_id,
                pending.tool_name,
            )

    async def _on_response_done(self, event: ResponseDoneEvent) -> None:
        if self._pending is None:
            return
        pending = self._pending
        if pending.supervisor_run.response_id == event.response_id:
            pending.supervisor_run.response_id = None
            pending.supervisor_run.response_done.set()

    async def _on_clarification_response(
        self, event: UserTranscriptCompletedEvent
    ) -> None:
        if self._pending is None:
            return
        future = self._pending.supervisor_run.pending_clarification_future
        if future and not future.done():
            logger.debug(
                "Clarification answered for '%s' [transcript=%s]",
                self._pending.tool_name,
                event.transcript,
            )
            future.set_result(event.transcript)
            self._pending.supervisor_run.pending_clarification_future = None

    async def _on_cancel_supervisor(self, _: CancelSupervisorCommand) -> None:
        if self._pending is None:
            return
        logger.info(
            "Cancelling supervisor task '%s' by request", self._pending.tool_name
        )
        self._cancel_pending(self._pending)
        self._pending = None
        await self._dispatch_busy_if_idle()
        await self._eject_cancel_tool()

    async def _deliver_result(self, pending: PendingToolCall) -> None:
        pending.channel_task = asyncio.create_task(self._channel_relay.run(pending))

        try:
            result = await pending.result_task
        except asyncio.CancelledError:
            logger.info("Tool '%s' was cancelled", pending.tool_name)
            if not pending.channel_task.done():
                pending.channel_task.cancel()
            return
        finally:
            if self._pending is pending:
                self._pending = None

        try:
            await self._await_channel_task(pending)

            serialized = self._ws.serialize(result)
            logger.info("Tool result: '%s' [result=%s]", pending.tool_name, serialized)
            await self._ws.send_function_call_output(pending.call_id, serialized)
            await self._ws.send_response_event(pending.tool)

            await self._dispatch_busy_if_idle()
            await self._eject_cancel_tool()
        except Exception:
            logger.exception("Failed to deliver result for '%s'", pending.tool_name)

    async def _await_channel_task(self, pending: PendingToolCall) -> None:
        if not pending.channel_task:
            return
        try:
            await pending.channel_task
        except Exception:
            logger.debug("Channel relay for '%s' ended with error", pending.tool_name)

    async def _inject_cancel_tool(self) -> None:
        if (
            self._cancel_tool
            and self._cancel_tool.name not in self._tools._registry.tools
        ):
            self._tools.inject_tool(self._cancel_tool)
            await self._dispatch_tools_update()
            logger.debug("Cancel tool '%s' injected", self._cancel_tool.name)

    async def _eject_cancel_tool(self) -> None:
        if self._cancel_tool:
            self._tools.eject_tool(self._cancel_tool.name)
            await self._dispatch_tools_update()
            logger.debug("Cancel tool '%s' ejected", self._cancel_tool.name)

    async def _create_supervisor_channel(self) -> SupervisorChannel:
        channel = SupervisorChannel()
        self._supervisor_agent._attach_channel(channel)
        await self._inject_cancel_tool()
        return channel

    def _cancel_pending(self, pending: PendingToolCall) -> None:
        future = pending.supervisor_run.pending_clarification_future
        if future and not future.done():
            future.cancel()
        if pending.channel_task and not pending.channel_task.done():
            pending.channel_task.cancel()
        pending.channel.cancel()
        pending.result_task.cancel()

    async def _dispatch_busy_if_idle(self) -> None:
        if self._pending is None:
            await self._event_bus.dispatch(SupervisorFinishedEvent())

    async def _dispatch_tools_update(self) -> None:
        await self._event_bus.dispatch(
            UpdateSessionToolsCommand(tools=self._tools.get_tool_schema())
        )

    async def _on_assistant_stopped(self, _: AssistantStoppedRespondingEvent) -> None:
        if self._pending is not None:
            self._pending.channel.notify_assistant_stopped()
