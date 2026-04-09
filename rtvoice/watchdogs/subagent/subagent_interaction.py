import asyncio
import json
import logging

from rtvoice.events import EventBus
from rtvoice.events.views import (
    AssistantStartedRespondingEvent,
    AssistantStoppedRespondingEvent,
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
from rtvoice.subagent.channel import SubAgentChannel
from rtvoice.subagent.views import SubAgentResult
from rtvoice.tools import Inject, Tools
from rtvoice.tools.registry.views import RealtimeTool
from rtvoice.watchdogs.subagent.views import PendingSubAgentCall
from rtvoice.watchdogs.tool_calling.helpers import ToolCallWebSocketHelper

logger = logging.getLogger(__name__)

_DEFAULT_HOLDING_INSTRUCTION = (
    "The user's request is being processed in the background. "
    "Say ONE very short, generic sentence (e.g. 'On it!', 'Give me a moment!', 'Sure, let me handle that!'). "
    "Do NOT mention what the task is. "
    "Do NOT state or imply whether it succeeded or failed — you don't know yet. "
    "Do NOT paraphrase the task. "
    "Always respond in the same language the user is speaking."
)


class SubAgentInteractionWatchdog:
    def __init__(
        self,
        event_bus: EventBus,
        tools: Tools,
        websocket: RealtimeWebSocket,
    ) -> None:
        self._event_bus = event_bus
        self._tools = tools
        self._ws = ToolCallWebSocketHelper(websocket)
        self._websocket = websocket
        self._cancel_tool = self._register_cancel_tool()
        self._active: PendingSubAgentCall | None = None
        self._subagents_by_tool_name: dict[str, SubAgent] = {}

        # When a subagent yields control back for clarification, only that
        # subagent is allowed to be called next — enforced in _handle_tool_call.
        self._awaiting_clarification_for: str | None = None

        self._response_idle = asyncio.Event()
        self._response_idle.set()

        self._event_bus.subscribe(FunctionCallItem, self._handle_tool_call)
        self._event_bus.subscribe(CancelSubAgentCommand, self._cancel_active_subagent)
        self._event_bus.subscribe(
            AssistantStartedRespondingEvent, self._on_response_started
        )
        self._event_bus.subscribe(
            AssistantStoppedRespondingEvent, self._on_response_stopped
        )

    def register_subagent(self, tool_name: str, agent: SubAgent) -> None:
        self._subagents_by_tool_name[tool_name] = agent
        logger.debug("Subagent '%s' registered on tool '%s'", agent.name, tool_name)

    def _register_cancel_tool(self) -> RealtimeTool:
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
            await self._ws.send_function_call_output(
                event.call_id, "Request already in progress."
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
            await self._ws.send_function_call_output(
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

        channel = await self._open_channel_for_status_updates(subagent)
        result_task = asyncio.create_task(
            self._tools.execute(event.name, event.arguments or {})
        )

        active = PendingSubAgentCall(
            call_id=event.call_id,
            subagent_name=event.name,
            execution_task=result_task,
            handoff_tool=tool,
            channel=channel,
        )
        self._active = active

        await self._event_bus.dispatch(SubAgentStartedEvent(agent_name=event.name))
        await self._send_holding_message(tool)
        asyncio.create_task(self._wait_for_result_and_respond(active))

    async def _send_holding_message(self, tool: RealtimeTool) -> None:
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
        active.channel_task = asyncio.create_task(self._stream_status_updates(active))

        try:
            result = await active.execution_task
        except asyncio.CancelledError:
            logger.info("Subagent '%s' was cancelled", active.subagent_name)
            if not active.channel_task.done():
                active.channel_task.cancel()
            return
        finally:
            if self._active is active:
                self._active = None

        try:
            await self._wait_for_status_updates_to_finish(active)
            await self._response_idle.wait()

            if isinstance(result, SubAgentResult) and result.clarification_needed:
                await self._ask_user_for_clarification(active, result)
                return

            self._awaiting_clarification_for = None
            serialized = self._ws.serialize(result)
            logger.info(
                "Subagent result: '%s' [result=%s]",
                active.subagent_name,
                serialized,
            )
            await self._ws.send_function_call_output(active.call_id, serialized)
            should_suppress_response = isinstance(result, SubAgentResult) and bool(
                result.suppress_realtime_response
            )

            if should_suppress_response:
                logger.info(
                    "Skipping realtime response inference for '%s' due to suppress_realtime_response",
                    active.subagent_name,
                )
            else:
                await self._ws.send_response_event(active.handoff_tool)

            await self._notify_subagent_finished(active.subagent_name)
            await self._eject_cancel_subagent_tool()
        except Exception:
            logger.exception("Failed to deliver result for '%s'", active.subagent_name)

    async def _ask_user_for_clarification(
        self, active: PendingSubAgentCall, result: SubAgentResult
    ) -> None:
        logger.info(
            "Subagent '%s' needs clarification: %s",
            active.subagent_name,
            result.clarification_needed,
        )

        # Lock the interaction to this subagent until the clarification is resolved.
        self._awaiting_clarification_for = active.subagent_name

        await self._ws.send_function_call_output(
            active.call_id,
            "Clarification needed before completing this task.",
        )
        await self._websocket.send(
            ConversationResponseCreateEvent.from_instructions(
                f'Ask the user: "{result.clarification_needed}". '
                f"Once they answer, call {active.subagent_name} again "
                f"with their answer in the `clarification_answer` field. "
                f"Do not make up the answer yourself.",
                tool_choice=ToolChoiceMode.AUTO,
            )
        )
        await self._notify_subagent_finished(active.subagent_name)
        await self._eject_cancel_subagent_tool()

    async def _wait_for_status_updates_to_finish(
        self, active: PendingSubAgentCall
    ) -> None:
        if not active.channel_task:
            return
        try:
            await active.channel_task
        except asyncio.CancelledError:
            logger.debug(
                "Status updates for '%s' were cancelled before finishing",
                active.subagent_name,
            )
        except Exception:
            logger.debug(
                "Status updates for '%s' ended with error", active.subagent_name
            )

    async def _stream_status_updates(self, active: PendingSubAgentCall) -> None:
        if not active.channel:
            return

        async for status in active.channel.events():
            logger.debug(
                "Status update for '%s': %s",
                active.subagent_name,
                status.message,
            )
            await self._websocket.send(
                ConversationResponseCreateEvent.from_instructions(
                    "Briefly summarise what was done in one short natural sentence (max 12 words). "
                    "If multiple steps are listed (separated by ->), combine them into one sentence. "
                    "Always respond in the same language the user is speaking. "
                    f"Steps: {status.message}",
                    tool_choice=ToolChoiceMode.NONE,
                )
            )

        logger.debug("All status updates sent for '%s'", active.subagent_name)

    async def _open_channel_for_status_updates(
        self, subagent: SubAgent
    ) -> SubAgentChannel:
        channel = SubAgentChannel()
        subagent._attach_channel(channel)
        await self._inject_cancel_subagent_tool()
        return channel

    def _abort_pending_call(self, active: PendingSubAgentCall) -> None:
        if active.channel_task and not active.channel_task.done():
            active.channel_task.cancel()
        active.channel.cancel()
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

    async def _on_response_started(self, _: AssistantStartedRespondingEvent) -> None:
        self._response_idle.clear()

    async def _on_response_stopped(self, _: AssistantStoppedRespondingEvent) -> None:
        self._response_idle.set()
        if self._active is not None:
            self._active.channel.notify_speech_ended()
