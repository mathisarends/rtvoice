import asyncio
import json
import logging

from rtvoice.events import EventBus
from rtvoice.events.views import (
    CancelSubAgentCommand,
    SubAgentFinishedEvent,
    SubAgentStartedEvent,
    UpdateSessionToolsCommand,
)
from rtvoice.realtime.schemas import (
    ConversationResponseCreateEvent,
    FunctionCallItem,
    ToolChoiceMode,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.subagent import SubAgent
from rtvoice.subagent.views import AgentClarificationNeeded
from rtvoice.tools import Inject, Tools
from rtvoice.tools.views import Tool
from rtvoice.watchdogs.subagent.views import PendingSubAgentCall
from rtvoice.watchdogs.tool_calling.helpers import (
    send_function_call_output,
    send_response_event,
    serialize_tool_result,
)

logger = logging.getLogger(__name__)

_DEFAULT_HOLDING_INSTRUCTION = (
    "The user's request is being processed in the background. "
    "Say ONE very short, generic sentence (e.g. 'On it!', 'Give me a moment!', 'Sure, let me handle that!'). "
    "Do NOT mention what the task is. "
    "Do NOT state or imply whether it succeeded or failed — you don't know yet. "
    "Do NOT paraphrase the task. "
    "Always respond in the same language the user is speaking."
)

_PROGRESS_PING_INSTRUCTION = (
    "The task is taking a bit longer. "
    "Say ONE short sentence reassuring the user you're still working on it. "
    "Always respond in the same language the user is speaking."
)


class SubAgentInteractionWatchdog:
    def __init__(
        self,
        event_bus: EventBus,
        tools: Tools,
        websocket: RealtimeWebSocket,
        progress_ping_delay: float = 8.0,
    ) -> None:
        self._event_bus = event_bus
        self._tools = tools
        self._websocket = websocket
        self._progress_ping_delay = progress_ping_delay
        self._cancel_tool = self._register_cancel_tool()
        self._active: PendingSubAgentCall | None = None
        self._subagents_by_tool_name: dict[str, SubAgent] = {}

        self._awaiting_clarification_for: str | None = None

        self._event_bus.subscribe(FunctionCallItem, self._handle_tool_call)
        self._event_bus.subscribe(CancelSubAgentCommand, self._cancel_active_subagent)

    def register_subagent(self, tool_name: str, agent: SubAgent) -> None:
        self._subagents_by_tool_name[tool_name] = agent
        logger.debug("Subagent '%s' registered on tool '%s'", agent.name, tool_name)

    def _register_cancel_tool(self) -> Tool:
        @self._tools.action(
            "Cancel the currently running background agent. "
            "Call this when the user explicitly wants to stop, cancel, or abandon the ongoing task.",
            name="cancel_agent",
            result_instruction="Tell the user naturally that the task has been cancelled.",
        )
        async def _cancel_agent(event_bus: Inject[EventBus]) -> str:
            await event_bus.dispatch(CancelSubAgentCommand())
            return "The agent task has been cancelled."

        tool = self._tools.get("cancel_agent")
        if not tool:
            raise RuntimeError("Failed to register cancel tool")
        return tool

    async def _handle_tool_call(self, event: FunctionCallItem) -> None:
        subagent = self._subagents_by_tool_name.get(event.name)
        if subagent is None:
            return

        tool = self._tools.get(event.name)
        if not tool:
            logger.error("Tool '%s' not found", event.name)
            return

        if self._active is not None:
            logger.warning(
                "Subagent already running, suppressing duplicate call for '%s'",
                event.name,
            )
            await send_function_call_output(
                self._websocket, event.call_id, "Request already in progress."
            )
            return

        if (
            self._awaiting_clarification_for is not None
            and event.name != self._awaiting_clarification_for
        ):
            logger.warning(
                "Subagent '%s' called while waiting for clarification answer for '%s' — rejecting.",
                event.name,
                self._awaiting_clarification_for,
            )
            await send_function_call_output(
                self._websocket,
                event.call_id,
                f"Cannot start a new task yet. Still waiting for the user's answer "
                f"to complete the '{self._awaiting_clarification_for}' task.",
            )
            return

        logger.info(
            "Subagent tool call: '%s' [args=%s]",
            event.name,
            json.dumps(event.arguments or {}, ensure_ascii=False),
        )

        await self._inject_cancel_subagent_tool()

        result_task = asyncio.create_task(
            self._tools.execute(event.name, event.arguments or {})
        )

        active = PendingSubAgentCall(
            call_id=event.call_id,
            subagent_name=event.name,
            execution_task=result_task,
            handoff_tool=tool,
        )
        self._active = active

        await self._event_bus.dispatch(SubAgentStartedEvent(agent_name=event.name))
        await self._send_holding_message(tool)
        asyncio.create_task(self._wait_for_result_and_respond(active))

    async def _send_holding_message(self, tool: Tool) -> None:
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                tool.holding_instruction or _DEFAULT_HOLDING_INSTRUCTION,
                tool_choice=ToolChoiceMode.NONE,
            )
        )

    async def _cancel_active_subagent(self, _: CancelSubAgentCommand) -> None:
        if self._active is None:
            return
        logger.info(
            "Cancelling subagent task '%s' by request", self._active.subagent_name
        )
        active_tool_name = self._active.subagent_name
        self._abort_pending_call(self._active)
        self._active = None
        self._awaiting_clarification_for = None
        await self._notify_subagent_finished(active_tool_name)
        await self._eject_cancel_subagent_tool()

    async def _wait_for_result_and_respond(self, active: PendingSubAgentCall) -> None:
        progress_task = asyncio.create_task(self._send_progress_ping_if_slow(active))

        try:
            result = await active.execution_task
        except asyncio.CancelledError:
            logger.info("Subagent '%s' was cancelled", active.subagent_name)
            return
        finally:
            progress_task.cancel()
            if self._active is active:
                self._active = None

        try:
            if isinstance(result, AgentClarificationNeeded):
                await self._ask_user_for_clarification(active, result)
                return

            self._awaiting_clarification_for = None
            serialized = serialize_tool_result(result)
            logger.info(
                "Subagent result: '%s' [result=%s]",
                active.subagent_name,
                serialized,
            )
            await send_function_call_output(self._websocket, active.call_id, serialized)
            await send_response_event(self._websocket, active.handoff_tool)
            await self._notify_subagent_finished(active.subagent_name)
            await self._eject_cancel_subagent_tool()
        except Exception:
            logger.exception("Failed to deliver result for '%s'", active.subagent_name)

    async def _send_progress_ping_if_slow(self, active: PendingSubAgentCall) -> None:
        try:
            await asyncio.sleep(self._progress_ping_delay)
        except asyncio.CancelledError:
            return

        if active.execution_task.done():
            return

        logger.debug("Sending progress ping for '%s'", active.subagent_name)
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                _PROGRESS_PING_INSTRUCTION,
                tool_choice=ToolChoiceMode.NONE,
            )
        )

    async def _ask_user_for_clarification(
        self, active: PendingSubAgentCall, result: AgentClarificationNeeded
    ) -> None:
        logger.info(
            "Subagent '%s' needs clarification: %s",
            active.subagent_name,
            result.question,
        )

        self._awaiting_clarification_for = active.subagent_name

        await send_function_call_output(
            self._websocket,
            active.call_id,
            "Clarification needed before completing this task.",
        )
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                f'Ask the user: "{result.question}". '
                f"Once they answer, call {active.subagent_name} again "
                f"with their answer in the `clarification_answer` field. "
                f"Do not make up the answer yourself.",
                tool_choice=ToolChoiceMode.AUTO,
            )
        )
        await self._notify_subagent_finished(active.subagent_name)
        await self._eject_cancel_subagent_tool()

    def _abort_pending_call(self, active: PendingSubAgentCall) -> None:
        active.execution_task.cancel()

    async def _notify_subagent_finished(self, agent_name: str) -> None:
        await self._event_bus.dispatch(SubAgentFinishedEvent(agent_name=agent_name))

    async def _inject_cancel_subagent_tool(self) -> None:
        if not self._tools.is_registered(self._cancel_tool):
            self._tools.inject_tool(self._cancel_tool)
            await self._sync_session_tools()
            logger.debug("Cancel tool '%s' injected", self._cancel_tool.name)

    async def _eject_cancel_subagent_tool(self) -> None:
        if self._cancel_tool:
            self._tools.eject_tool(self._cancel_tool.name)
            await self._sync_session_tools()
            logger.debug("Cancel tool '%s' ejected", self._cancel_tool.name)

    async def _sync_session_tools(self) -> None:
        await self._event_bus.dispatch(
            UpdateSessionToolsCommand(tools=self._tools.get_tool_schema())
        )
