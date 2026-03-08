from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from rtvoice.events import EventBus
from rtvoice.events.views import AssistantInterruptedEvent, UserTranscriptCompletedEvent
from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
    FunctionCallItem,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ToolChoiceMode,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.supervisor.channel import StatusMessage, SupervisorChannel, UserQuestion
from rtvoice.tools import Tools
from rtvoice.tools.registry.views import Tool

if TYPE_CHECKING:
    from rtvoice.supervisor import SupervisorAgent

logger = logging.getLogger(__name__)

_DEFAULT_HOLDING_INSTRUCTION = (
    "The user's request is being processed. "
    "Say ONE brief sentence acknowledging it naturally "
    "(e.g. 'Let me look that up for you!'). "
    "Then stop. Do not keep talking and do not reveal any results."
)


@dataclass
class _PendingToolCall:
    call_id: str
    tool_name: str
    result_task: asyncio.Task
    tool: Tool
    holding_response_id: str | None = None
    holding_done: asyncio.Event = field(default_factory=asyncio.Event)
    channel: SupervisorChannel | None = None
    channel_task: asyncio.Task | None = None
    pending_clarification_future: asyncio.Future[str] | None = None


class ToolCallingWatchdog:
    def __init__(self, event_bus: EventBus, tools: Tools, websocket: RealtimeWebSocket):
        self._event_bus = event_bus
        self._tools = tools
        self._websocket = websocket
        self._pending: list[_PendingToolCall] = []
        self._supervisor_agents: dict[str, SupervisorAgent] = {}

        self._event_bus.subscribe(FunctionCallItem, self._handle_tool_call)
        self._event_bus.subscribe(ResponseCreatedEvent, self._on_response_created)
        self._event_bus.subscribe(ResponseDoneEvent, self._on_response_done)
        self._event_bus.subscribe(
            UserTranscriptCompletedEvent, self._on_user_transcript
        )
        self._event_bus.subscribe(AssistantInterruptedEvent, self._on_interrupted)

    def register_supervisor(self, tool_name: str, agent: SupervisorAgent) -> None:
        """Register a supervisor agent so the watchdog can attach a channel to its runs."""
        self._supervisor_agents[tool_name] = agent

    async def _handle_tool_call(self, event: FunctionCallItem) -> None:
        tool = self._tools.get(event.name)
        if not tool:
            logger.error("Tool '%s' not found", event.name)
            return

        if tool.is_long_running:
            existing = next(
                (p for p in self._pending if p.tool_name == event.name), None
            )
            if existing:
                logger.warning(
                    "Duplicate tool call for '%s' while already pending, suppressing",
                    event.name,
                )
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

    async def _handle_long_running(self, event: FunctionCallItem, tool: Tool) -> None:
        channel: SupervisorChannel | None = None
        supervisor = self._supervisor_agents.get(event.name)
        if supervisor:
            channel = SupervisorChannel()
            supervisor._attach_channel(channel)

        result_task = asyncio.create_task(
            self._tools.execute(event.name, event.arguments or {})
        )

        pending = _PendingToolCall(
            call_id=event.call_id,
            tool_name=event.name,
            result_task=result_task,
            tool=tool,
            channel=channel,
        )
        self._pending.append(pending)

        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                tool.holding_instruction or _DEFAULT_HOLDING_INSTRUCTION,
                tool_choice=ToolChoiceMode.NONE,
            )
        )

        asyncio.create_task(self._deliver_result(pending))

    async def _deliver_result(self, pending: _PendingToolCall) -> None:
        await pending.holding_done.wait()

        if pending.channel:
            pending.channel_task = asyncio.create_task(self._process_channel(pending))

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

    async def _process_channel(self, pending: _PendingToolCall) -> None:
        """Relay supervisor channel events to the user via the RealtimeAgent."""
        async for event in pending.channel.events():
            if isinstance(event, StatusMessage):
                logger.debug(
                    "Supervisor status for '%s': %s", pending.tool_name, event.message
                )
                await self._websocket.send(
                    ConversationResponseCreateEvent.from_instructions(
                        f"Briefly summarise what was done in one short natural sentence (max 12 words). "
                        f"If multiple steps are listed (separated by →), combine them into one sentence. "
                        f"Steps: {event.message}",
                        tool_choice=ToolChoiceMode.NONE,
                    )
                )
            elif isinstance(event, UserQuestion):
                logger.debug(
                    "Supervisor clarification for '%s': %s",
                    pending.tool_name,
                    event.question,
                )
                pending.pending_clarification_future = event.answer_future
                await self._websocket.send(
                    ConversationResponseCreateEvent.from_instructions(
                        f'Ask the user naturally and conversationally: "{event.question}"',
                        tool_choice=ToolChoiceMode.NONE,
                    )
                )

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
                break

    async def _on_response_done(self, event: ResponseDoneEvent) -> None:
        for pending in self._pending:
            if pending.holding_response_id == event.response_id:
                pending.holding_done.set()
                logger.debug("Holding done for '%s'", pending.tool_name)
                break

    async def _on_user_transcript(self, event: UserTranscriptCompletedEvent) -> None:
        for pending in self._pending:
            if (
                pending.pending_clarification_future
                and not pending.pending_clarification_future.done()
            ):
                logger.debug(
                    "Clarification answered for '%s': %s",
                    pending.tool_name,
                    event.transcript,
                )
                pending.pending_clarification_future.set_result(event.transcript)
                pending.pending_clarification_future = None
                return

    async def _on_interrupted(self, _: AssistantInterruptedEvent) -> None:
        if not self._pending:
            return
        logger.info(
            "Interruption detected — cancelling %d pending tool(s)", len(self._pending)
        )
        for pending in self._pending:
            if pending.channel:
                pending.channel.cancel()
            if pending.channel_task and not pending.channel_task.done():
                pending.channel_task.cancel()
            pending.result_task.cancel()
        self._pending.clear()

    def _serialize(self, result: Any) -> str:
        if result is None:
            return "Success"
        if isinstance(result, str):
            return result
        if isinstance(result, BaseModel):
            return result.model_dump_json(exclude_none=True)
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
