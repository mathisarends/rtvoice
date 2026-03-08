from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentBusyEvent,
    AssistantInterruptedEvent,
    CancelSupervisorCommand,
    UpdateSessionToolsCommand,
    UserTranscriptCompletedEvent,
)
from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
    FunctionCallItem,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ToolChoiceMode,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.supervisor.channel import SupervisorChannel
from rtvoice.tools import Tools
from rtvoice.tools.registry.views import Tool
from rtvoice.tools.views import VoidResult
from rtvoice.watchdogs.tool_calling.channel_relay import ChannelRelay
from rtvoice.watchdogs.tool_calling.views import PendingSupervisorRun, PendingToolCall

if TYPE_CHECKING:
    from rtvoice.supervisor import SupervisorAgent

logger = logging.getLogger(__name__)

_DEFAULT_HOLDING_INSTRUCTION = (
    "The user's request is being processed. "
    "Say ONE brief sentence acknowledging it naturally "
    "(e.g. 'Let me look that up for you!'). "
    "Then stop. Do not keep talking and do not reveal any results."
)


# TODO: Refactor this
class ToolCallingWatchdog:
    def __init__(
        self,
        event_bus: EventBus,
        tools: Tools,
        websocket: RealtimeWebSocket,
        cancel_tool: Tool | None = None,
    ):
        self._event_bus = event_bus
        self._tools = tools
        self._websocket = websocket
        self._pending: list[PendingToolCall] = []
        self._supervisor_agents: dict[str, SupervisorAgent] = {}
        self._channel_relay = ChannelRelay(websocket)
        self._cancel_tool = cancel_tool

        self._event_bus.subscribe(FunctionCallItem, self._handle_tool_call)
        self._event_bus.subscribe(ResponseCreatedEvent, self._on_response_created)
        self._event_bus.subscribe(ResponseDoneEvent, self._on_response_done)
        self._event_bus.subscribe(
            UserTranscriptCompletedEvent, self._on_clarification_response
        )
        self._event_bus.subscribe(AssistantInterruptedEvent, self._on_interrupted)
        self._event_bus.subscribe(CancelSupervisorCommand, self._on_cancel_supervisor)

    def register_supervisor(self, tool_name: str, agent: SupervisorAgent) -> None:
        self._supervisor_agents[tool_name] = agent

    async def _handle_tool_call(self, event: FunctionCallItem) -> None:
        tool = self._tools.get(event.name)
        if not tool:
            logger.error("Tool '%s' not found", event.name)
            return

        is_already_pending = any(p.tool_name == event.name for p in self._pending)
        if tool.is_long_running and is_already_pending:
            logger.warning("Duplicate tool call for '%s', suppressing", event.name)
            await self._websocket.send(
                ConversationItemCreateEvent.function_call_output(
                    call_id=event.call_id,
                    output="Request already in progress.",
                )
            )
            return

        logger.info(
            "Tool call: '%s' | args: %s",
            event.name,
            json.dumps(event.arguments or {}, ensure_ascii=False),
        )

        if tool.is_long_running:
            await self._handle_long_running(event, tool)
        else:
            await self._handle_immediate(event, tool)

    async def _eject_cancel_tool(self) -> None:
        if self._cancel_tool:
            self._tools.eject_tool(self._cancel_tool.name)
            await self._event_bus.dispatch(
                UpdateSessionToolsCommand(tools=self._tools.get_tool_schema())
            )

    async def _handle_long_running(self, event: FunctionCallItem, tool: Tool) -> None:
        channel: SupervisorChannel | None = None
        supervisor = self._supervisor_agents.get(event.name)
        if supervisor:
            channel = SupervisorChannel()
            supervisor._attach_channel(channel)
            await self._inject_cancel_tool()

        result_task = asyncio.create_task(
            self._tools.execute(event.name, event.arguments or {})
        )

        pending = PendingToolCall(
            call_id=event.call_id,
            tool_name=event.name,
            result_task=result_task,
            tool=tool,
            channel=channel,
            supervisor_run=PendingSupervisorRun() if channel else None,
        )
        self._pending.append(pending)
        await self._event_bus.dispatch(AgentBusyEvent(busy=True))

        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                tool.holding_instruction or _DEFAULT_HOLDING_INSTRUCTION,
                tool_choice=ToolChoiceMode.NONE,
            )
        )
        asyncio.create_task(self._deliver_result(pending))

    async def _inject_cancel_tool(self) -> None:
        if (
            self._cancel_tool
            and self._cancel_tool.name not in self._tools._registry.tools
        ):
            self._tools.inject_tool(self._cancel_tool)
            await self._event_bus.dispatch(
                UpdateSessionToolsCommand(tools=self._tools.get_tool_schema())
            )

    async def _deliver_result(self, pending: PendingToolCall) -> None:
        await pending.holding_done.wait()

        if pending.channel:
            pending.channel_task = asyncio.create_task(self._channel_relay.run(pending))

        try:
            result = await pending.result_task
        except asyncio.CancelledError:
            logger.info("Tool '%s' was cancelled", pending.tool_name)
            if pending.channel_task and not pending.channel_task.done():
                pending.channel_task.cancel()
            return
        finally:
            if pending in self._pending:
                self._pending.remove(pending)

        if pending.channel_task:
            try:
                await pending.channel_task
            except Exception:
                logger.debug(
                    "Channel relay for '%s' ended with error", pending.tool_name
                )

        logger.info(
            "Tool result: '%s' | %s", pending.tool_name, self._serialize(result)
        )

        await self._websocket.send(
            ConversationItemCreateEvent.function_call_output(
                call_id=pending.call_id,
                output=self._serialize(result),
            )
        )
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                pending.tool.result_instruction
            )
            if pending.tool.result_instruction
            else ConversationResponseCreateEvent()
        )

        if not self._pending:
            await self._event_bus.dispatch(AgentBusyEvent(busy=False))

        if pending.channel and not any(p.channel for p in self._pending):
            await self._eject_cancel_tool()

    async def _handle_immediate(self, event: FunctionCallItem, tool: Tool) -> None:
        result = await self._tools.execute(event.name, event.arguments or {})
        logger.info("Tool result: '%s' | %s", event.name, self._serialize(result))

        await self._websocket.send(
            ConversationItemCreateEvent.function_call_output(
                call_id=event.call_id,
                output=self._serialize(result),
            )
        )
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(tool.result_instruction)
            if tool.result_instruction
            else ConversationResponseCreateEvent()
        )

    async def _on_response_created(self, event: ResponseCreatedEvent) -> None:
        for pending in self._pending:
            if pending.holding_response_id is None:
                pending.holding_response_id = event.response_id
                logger.debug(
                    "Holding response '%s' tracked for '%s'",
                    event.response_id,
                    pending.tool_name,
                )
                return
            if pending.supervisor_run and pending.supervisor_run.response_id is None:
                pending.supervisor_run.response_id = event.response_id
                logger.debug(
                    "Status response '%s' tracked for '%s'",
                    event.response_id,
                    pending.tool_name,
                )
                return

    async def _on_response_done(self, event: ResponseDoneEvent) -> None:
        for pending in self._pending:
            if pending.holding_response_id == event.response_id:
                pending.holding_done.set()
                logger.debug("Holding done for '%s'", pending.tool_name)
                return
            if (
                pending.supervisor_run
                and pending.supervisor_run.response_id == event.response_id
            ):
                pending.supervisor_run.response_id = None
                pending.supervisor_run.response_done.set()
                return

    async def _on_clarification_response(
        self, event: UserTranscriptCompletedEvent
    ) -> None:
        for pending in self._pending:
            if (
                pending.supervisor_run
                and pending.supervisor_run.pending_clarification_future
                and not pending.supervisor_run.pending_clarification_future.done()
            ):
                logger.debug(
                    "Clarification answered for '%s': %s",
                    pending.tool_name,
                    event.transcript,
                )
                pending.supervisor_run.pending_clarification_future.set_result(
                    event.transcript
                )
                pending.supervisor_run.pending_clarification_future = None

    async def _on_interrupted(self, _: AssistantInterruptedEvent) -> None:
        non_supervisor = [p for p in self._pending if not p.channel]
        if not non_supervisor:
            return
        logger.info(
            "Interruption — cancelling %d non-supervisor tool(s)", len(non_supervisor)
        )
        for pending in non_supervisor:
            pending.result_task.cancel()
            self._pending.remove(pending)
        if not self._pending:
            await self._event_bus.dispatch(AgentBusyEvent(busy=False))

    async def _on_cancel_supervisor(self, _: CancelSupervisorCommand) -> None:
        supervisor_tasks = [p for p in self._pending if p.channel]
        if not supervisor_tasks:
            return
        logger.info(
            "Cancelling %d supervisor task(s) by request", len(supervisor_tasks)
        )
        for pending in supervisor_tasks:
            if (
                pending.supervisor_run
                and pending.supervisor_run.pending_clarification_future
                and not pending.supervisor_run.pending_clarification_future.done()
            ):
                pending.supervisor_run.pending_clarification_future.cancel()
            if pending.channel_task and not pending.channel_task.done():
                pending.channel_task.cancel()
            pending.channel.cancel()
            pending.result_task.cancel()
            self._pending.remove(pending)
        if not self._pending:
            await self._event_bus.dispatch(AgentBusyEvent(busy=False))
        await self._eject_cancel_tool()

    def _serialize(self, result: Any) -> str:
        if isinstance(result, VoidResult):
            return str(result)
        if isinstance(result, str):
            return result
        if isinstance(result, BaseModel):
            return result.model_dump_json(exclude_none=True)
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
