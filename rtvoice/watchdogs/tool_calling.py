import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from rtvoice.events import EventBus
from rtvoice.events.views import UserTranscriptCompletedEvent
from rtvoice.realtime.schemas import (
    ConversationItemCreateEvent,
    ConversationResponseCreateEvent,
    FunctionCallItem,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ToolChoiceMode,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.subagents.views import SubAgentClarificationNeeded
from rtvoice.tools import Tools
from rtvoice.tools.registry.views import Tool

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


class ToolCallingWatchdog:
    def __init__(self, event_bus: EventBus, tools: Tools, websocket: RealtimeWebSocket):
        self._event_bus = event_bus
        self._tools = tools
        self._websocket = websocket
        self._pending: list[_PendingToolCall] = []
        self._pending_clarification: SubAgentClarificationNeeded | None = None

        self._event_bus.subscribe(FunctionCallItem, self._handle_tool_call)
        self._event_bus.subscribe(ResponseCreatedEvent, self._on_response_created)
        self._event_bus.subscribe(ResponseDoneEvent, self._on_response_done)
        self._event_bus.subscribe(
            SubAgentClarificationNeeded, self._on_clarification_needed
        )
        self._event_bus.subscribe(
            UserTranscriptCompletedEvent, self._on_user_transcript
        )

    async def _handle_tool_call(self, event: FunctionCallItem) -> None:
        tool = self._tools.get(event.name)
        if not tool:
            logger.error("Tool '%s' not found", event.name)
            return

        logger.info(
            "Tool call started: '%s' | args: %s",
            event.name,
            json.dumps(event.arguments or {}, ensure_ascii=False),
        )

        if tool.is_long_running:
            await self._handle_long_running(event, tool)
        else:
            await self._handle_immediate(event, tool)

    async def _handle_long_running(self, event: FunctionCallItem, tool: Tool) -> None:
        result_task = asyncio.create_task(
            self._tools.execute(event.name, event.arguments or {})
        )

        pending = _PendingToolCall(
            call_id=event.call_id,
            tool_name=event.name,
            result_task=result_task,
            tool=tool,
        )
        self._pending.append(pending)

        holding_instruction = tool.holding_instruction or _DEFAULT_HOLDING_INSTRUCTION
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                holding_instruction,
                tool_choice=ToolChoiceMode.NONE,
            )
        )

        asyncio.create_task(self._deliver_result(pending))

    async def _deliver_result(self, pending: _PendingToolCall) -> None:
        await pending.holding_done.wait()
        result = await pending.result_task

        logger.info(
            "Tool call result: '%s' | result: %s",
            pending.tool_name,
            self._serialize(result),
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

    async def _handle_immediate(self, event: FunctionCallItem, tool: Tool) -> None:
        result = await self._tools.execute(event.name, event.arguments or {})

        logger.info(
            "Tool call result: '%s' | result: %s",
            event.name,
            self._serialize(result),
        )

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
                    "Holding response '%s' tracked for tool '%s'",
                    event.response_id,
                    pending.tool_name,
                )
                break

    async def _on_response_done(self, event: ResponseDoneEvent) -> None:
        for pending in self._pending:
            if pending.holding_response_id == event.response_id:
                pending.holding_done.set()
                self._pending.remove(pending)
                logger.debug(
                    "Holding response '%s' done – unblocking result delivery for '%s'",
                    event.response_id,
                    pending.tool_name,
                )
                break

    async def _on_clarification_needed(
        self, event: SubAgentClarificationNeeded
    ) -> None:
        self._pending_clarification = event
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                f"Ask the user the following question naturally and conversationally: "
                f'"{event.question}"'
            )
        )

    async def _on_user_transcript(self, event: UserTranscriptCompletedEvent) -> None:
        if self._pending_clarification is None:
            return

        clarification = self._pending_clarification
        self._pending_clarification = None
        clarification.answer_future.set_result(event.transcript)

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
