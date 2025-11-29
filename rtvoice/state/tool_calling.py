from rtvoice.state.base import AssistantState, VoiceAssistantEvent
from rtvoice.state.context import VoiceAssistantContext
from rtvoice.state.models import StateType


class ToolCallingState(AssistantState):
    def __init__(self):
        self._event_handlers = {
            VoiceAssistantEvent.ASSISTANT_RECEIVED_TOOL_CALL_RESULT: self._handle_tool_call_result,
            VoiceAssistantEvent.ASSISTANT_COMPLETED_MCP_TOOL_CALL_RESULT: self._handle_mcp_tool_call_completed,
            VoiceAssistantEvent.ASSISTANT_FAILED_MCP_TOOL_CALL: self._handle_mcp_tool_call_failed,
        }

    @property
    def state_type(self) -> StateType:
        return StateType.TOOL_CALLING

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        handler = self._event_handlers.get(event)
        if handler:
            await handler(context)

    async def _handle_tool_call_result(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Tool call result received")
        await self._transition_to_responding(context)

    async def _handle_mcp_tool_call_completed(
        self, context: VoiceAssistantContext
    ) -> None:
        self.logger.info("MCP tool call completed - transitioning to Responding state")
        await self._transition_to_responding(context)

    async def _handle_mcp_tool_call_failed(
        self, context: VoiceAssistantContext
    ) -> None:
        self.logger.info("MCP tool call failed - transitioning to Responding state")
        await self._transition_to_responding(context)
