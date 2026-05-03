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
    UpdateSupervisorCommand,
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


@dataclass
class PendingSupervisorCall:
    call_id: str
    handoff_tool: Tool
    arguments: dict
    started: asyncio.Event
    execution_task: asyncio.Task | None = None
    runner_task: asyncio.Task | None = None


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
        self._update_tool = self._register_update_tool()
        self._tools.eject_tool(self._cancel_tool.name)
        self._tools.eject_tool(self._update_tool.name)
        self._active: PendingSupervisorCall | None = None
        self._awaiting_clarification = False

        self._event_bus.subscribe(FunctionCallItem, self._handle_tool_call)
        self._event_bus.subscribe(
            CancelSupervisorCommand, self._cancel_active_supervisor
        )
        self._event_bus.subscribe(
            UpdateSupervisorCommand, self._update_active_supervisor
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

    def _register_update_tool(self) -> Tool:
        @self._tools.action(
            "Send new context, corrections, or additional instructions to the currently running supervisor. "
            "Call this when the user adds information while the supervisor is still working, instead of restarting the task.",
            name="update_supervisor",
            result_instruction="Briefly acknowledge that the update was added to the running task.",
        )
        async def _update_supervisor(message: str, event_bus: Inject[EventBus]) -> str:
            await event_bus.dispatch(UpdateSupervisorCommand(message=message))
            return "The supervisor has received the update."

        tool = self._tools.get("update_supervisor")
        if not tool:
            raise RuntimeError("Failed to register update tool")
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

        await self._inject_supervisor_control_tools()
        self._supervisor.on_progress = self._send_progress_update

        active = PendingSupervisorCall(
            call_id=event.call_id,
            handoff_tool=tool,
            arguments=event.arguments or {},
            started=asyncio.Event(),
        )
        self._active = active

        await self._event_bus.dispatch(SupervisorStartedEvent())
        await self._send_holding_message(tool)
        active.runner_task = asyncio.create_task(self._run_supervisor_call(active))
        active.runner_task.add_done_callback(self._log_unhandled_runner_exception)
        await active.started.wait()

    async def _send_progress_update(self, message: str) -> None:
        logger.debug("Sending supervisor progress update: %s", message)
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                f'Tell the user briefly: "{message}". One sentence, same language.',
                tool_choice=ToolChoiceMode.NONE,
            )
        )

    async def _send_holding_message(self, tool: Tool) -> None:
        if not tool.holding_instruction:
            return

        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                tool.holding_instruction,
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
        await self._eject_supervisor_control_tools()

    async def _update_active_supervisor(self, event: UpdateSupervisorCommand) -> None:
        if self._active is None:
            logger.debug("Ignoring supervisor update because no task is active")
            return

        logger.info("Updating active supervisor task: %s", event.message)
        await self._supervisor.update(event.message)

    async def _run_supervisor_call(self, active: PendingSupervisorCall) -> None:
        failure_message: str | None = None

        try:
            try:
                async with asyncio.TaskGroup() as task_group:
                    active.execution_task = task_group.create_task(
                        self._tools.execute(active.handoff_tool.name, active.arguments)
                    )
                    active.started.set()
            except* Exception as exception_group:
                failure_message = str(exception_group.exceptions[0])
        except asyncio.CancelledError:
            logger.info("Supervisor was cancelled")
            return
        finally:
            if self._active is active:
                self._active = None

        if failure_message is not None:
            logger.error("Supervisor failed: %s", failure_message)
            await self._handle_supervisor_failure(active, failure_message)
            return

        if active.execution_task is None or active.execution_task.cancelled():
            logger.info("Supervisor was cancelled")
            return

        result = active.execution_task.result()
        await self._deliver_supervisor_result(active, result)

    async def _deliver_supervisor_result(
        self, active: PendingSupervisorCall, result: object
    ) -> None:
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
            await self._eject_supervisor_control_tools()
        except Exception:
            logger.exception("Failed to deliver supervisor result")

    async def _handle_supervisor_failure(
        self, active: PendingSupervisorCall, message: str
    ) -> None:
        await send_function_call_output(
            self._websocket,
            active.call_id,
            f"Supervisor task failed: {message}",
        )
        await self._notify_supervisor_finished()
        await self._eject_supervisor_control_tools()

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
        await self._eject_supervisor_control_tools()

    def _abort_pending_call(self, active: PendingSupervisorCall) -> None:
        if active.execution_task is not None:
            active.execution_task.cancel()
            return
        if active.runner_task is not None:
            active.runner_task.cancel()

    def _log_unhandled_runner_exception(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            logger.exception("Supervisor runner failed", exc_info=exception)

    async def _notify_supervisor_finished(self) -> None:
        await self._event_bus.dispatch(SupervisorFinishedEvent())

    async def _inject_supervisor_control_tools(self) -> None:
        changed = False
        for tool in (self._cancel_tool, self._update_tool):
            if not self._tools.is_registered(tool):
                self._tools.inject_tool(tool)
                changed = True
                logger.debug("Supervisor control tool '%s' injected", tool.name)

        if changed:
            await self._sync_session_tools()

    async def _eject_supervisor_control_tools(self) -> None:
        changed = False
        for tool in (self._cancel_tool, self._update_tool):
            if self._tools.get(tool.name) is not None:
                self._tools.eject_tool(tool.name)
                changed = True
                logger.debug("Supervisor control tool '%s' ejected", tool.name)

        self._supervisor.discard_pending_updates()

        if changed:
            await self._sync_session_tools()

    async def _sync_session_tools(self) -> None:
        await self._event_bus.dispatch(
            UpdateSessionToolsCommand(tools=self._tools.get_tool_schema())
        )
