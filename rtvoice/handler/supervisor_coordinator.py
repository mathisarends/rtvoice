from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rtvoice.agent.views import SupervisorClarificationNeeded
from rtvoice.events import EventBus
from rtvoice.events.views import (
    CancelSupervisorCommand,
    SupervisorFinishedEvent,
    SupervisorStartedEvent,
    UpdateSessionToolsCommand,
)
from rtvoice.handler.tool_call_helpers import (
    send_function_call_output,
    send_response_event,
    serialize_tool_result,
)
from rtvoice.realtime.schemas import (
    ConversationResponseCreateEvent,
    FunctionCallItem,
    ToolChoiceMode,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.tools.di import Inject
from rtvoice.tools.views import Tool

if TYPE_CHECKING:
    from rtvoice.agent.supervisor import Supervisor
    from rtvoice.tools import Tools

logger = logging.getLogger(__name__)

_DEFAULT_HOLDING_INSTRUCTION = (
    "The user's request is being processed in the background. "
    "Say ONE very short, generic sentence (e.g. 'On it!', 'Give me a moment!', 'Sure, let me handle that!'). "
    "Do NOT mention what the task is. "
    "Do NOT state or imply whether it succeeded or failed - you don't know yet. "
    "Do NOT paraphrase the task. "
    "Always respond in the same language the user is speaking."
)


@dataclass
class PendingSupervisorCall:
    call_id: str
    execution_task: asyncio.Task
    handoff_tool: Tool


class SupervisorCoordinator:
    def __init__(
        self,
        event_bus: EventBus,
        tools: Tools,
        websocket: RealtimeWebSocket,
        supervisor: Supervisor,
    ) -> None:
        self._event_bus = event_bus
        self._tools = tools
        self._websocket = websocket
        self._supervisor = supervisor
        self._cancel_tool = self._register_cancel_tool()
        self._active: PendingSupervisorCall | None = None
        self._awaiting_clarification = False

        self._event_bus.subscribe(FunctionCallItem, self._handle_tool_call)
        self._event_bus.subscribe(
            CancelSupervisorCommand, self._cancel_active_supervisor
        )

    def _register_cancel_tool(self) -> Tool:
        @self._tools.action(
            "Cancel the currently running supervisor. "
            "Call this when the user explicitly wants to stop, cancel, or abandon the ongoing task.",
            name="cancel_supervisor",
            result_instruction="Tell the user naturally that the task has been cancelled.",
        )
        async def _cancel_supervisor(event_bus: Inject[EventBus]) -> str:
            await event_bus.dispatch(CancelSupervisorCommand())
            return "The supervisor task has been cancelled."

        tool = self._tools.get("cancel_supervisor")
        if not tool:
            raise RuntimeError("Failed to register cancel tool")
        return tool

    async def _handle_tool_call(self, event: FunctionCallItem) -> None:
        if event.name != self._supervisor.name:
            return

        tool = self._tools.get(event.name)
        if not tool:
            logger.error("Tool '%s' not found", event.name)
            return

        if self._active is not None:
            logger.warning("Supervisor already running, suppressing duplicate call")
            await send_function_call_output(
                self._websocket, event.call_id, "Request already in progress."
            )
            return

        if self._awaiting_clarification and not (event.arguments or {}).get(
            "clarification_answer"
        ):
            logger.warning(
                "Supervisor called without clarification answer while paused"
            )
            await send_function_call_output(
                self._websocket,
                event.call_id,
                "Cannot continue yet. Still waiting for the user's clarification answer.",
            )
            return

        logger.info(
            "Supervisor tool call [args=%s]",
            json.dumps(event.arguments or {}, ensure_ascii=False),
        )

        await self._inject_cancel_supervisor_tool()
        self._supervisor.on_progress = self._send_progress_update

        result_task = asyncio.create_task(
            self._tools.execute(event.name, event.arguments or {})
        )

        active = PendingSupervisorCall(
            call_id=event.call_id,
            execution_task=result_task,
            handoff_tool=tool,
        )
        self._active = active

        await self._event_bus.dispatch(SupervisorStartedEvent())
        await self._send_holding_message(tool)
        asyncio.create_task(self._wait_for_result_and_respond(active))

    async def _send_progress_update(self, message: str) -> None:
        logger.debug("Sending supervisor progress update: %s", message)
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                f'Tell the user briefly: "{message}". One sentence, same language.',
                tool_choice=ToolChoiceMode.NONE,
            )
        )

    async def _send_holding_message(self, tool: Tool) -> None:
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                tool.holding_instruction or _DEFAULT_HOLDING_INSTRUCTION,
                tool_choice=ToolChoiceMode.NONE,
            )
        )

    async def _cancel_active_supervisor(self, _: CancelSupervisorCommand) -> None:
        if self._active is None:
            return
        logger.info("Cancelling supervisor task by request")
        self._abort_pending_call(self._active)
        self._active = None
        self._awaiting_clarification = False
        await self._notify_supervisor_finished()
        await self._eject_cancel_supervisor_tool()

    async def _wait_for_result_and_respond(self, active: PendingSupervisorCall) -> None:
        try:
            result = await active.execution_task
        except asyncio.CancelledError:
            logger.info("Supervisor was cancelled")
            return
        finally:
            if self._active is active:
                self._active = None

        try:
            if isinstance(result, SupervisorClarificationNeeded):
                await self._ask_user_for_clarification(active, result)
                return

            self._awaiting_clarification = False
            serialized = serialize_tool_result(result)
            logger.info("Supervisor result [result=%s]", serialized)
            await send_function_call_output(self._websocket, active.call_id, serialized)
            await send_response_event(self._websocket, active.handoff_tool)
            await self._notify_supervisor_finished()
            await self._eject_cancel_supervisor_tool()
        except Exception:
            logger.exception("Failed to deliver supervisor result")

    async def _ask_user_for_clarification(
        self, active: PendingSupervisorCall, result: SupervisorClarificationNeeded
    ) -> None:
        logger.info("Supervisor needs clarification: %s", result.question)

        self._awaiting_clarification = True

        await send_function_call_output(
            self._websocket,
            active.call_id,
            "Clarification needed before completing this task.",
        )
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                f'Ask the user: "{result.question}". '
                f"Once they answer, call {self._supervisor.name} again "
                f"with their answer in the `clarification_answer` field. "
                f"Do not make up the answer yourself.",
                tool_choice=ToolChoiceMode.AUTO,
            )
        )
        await self._notify_supervisor_finished()
        await self._eject_cancel_supervisor_tool()

    def _abort_pending_call(self, active: PendingSupervisorCall) -> None:
        active.execution_task.cancel()

    async def _notify_supervisor_finished(self) -> None:
        await self._event_bus.dispatch(SupervisorFinishedEvent())

    async def _inject_cancel_supervisor_tool(self) -> None:
        if not self._tools.is_registered(self._cancel_tool):
            self._tools.inject_tool(self._cancel_tool)
            await self._sync_session_tools()
            logger.debug("Cancel tool '%s' injected", self._cancel_tool.name)

    async def _eject_cancel_supervisor_tool(self) -> None:
        if self._cancel_tool:
            self._tools.eject_tool(self._cancel_tool.name)
            await self._sync_session_tools()
            logger.debug("Cancel tool '%s' ejected", self._cancel_tool.name)

    async def _sync_session_tools(self) -> None:
        await self._event_bus.dispatch(
            UpdateSessionToolsCommand(tools=self._tools.get_tool_schema())
        )
