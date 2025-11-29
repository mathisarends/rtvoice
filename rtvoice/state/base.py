from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from rtvoice.shared.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from rtvoice.state.context import VoiceAssistantContext


class VoiceAssistantEvent(Enum):
    WAKE_WORD_DETECTED = "wake_word_detected"
    USER_STARTED_SPEAKING = "user_started_speaking"
    USER_SPEECH_ENDED = "user_speech_ended"

    USER_TRANSCRIPT_CHUNK_RECEIVED = "user_transcript_chunk_received"
    USER_TRANSCRIPT_COMPLETED = "user_transcript_completed"

    AUDIO_CHUNK_RECEIVED = "audio_chunk_received"

    ASSISTANT_STARTED_RESPONDING = "assistant_started_responding"
    ASSISTANT_RESPONSE_COMPLETED = "assistant_response_completed"
    ASSISTANT_SPEECH_INTERRUPTED = "assistant_speech_interrupted"

    # these events have nothing to to with the audio streaming, but with the data which is sent and received
    ASSISTANT_STARTED_RESPONSE = "assistant_started_response"
    ASSISTANT_COMPLETED_RESPONSE = "assistant_completed_response"

    ASSISTANT_TRANSCRIPT_CHUNK_RECEIVED = "assistant_transcript_chunk_received"
    ASSISTANT_TRANSCRIPT_COMPLETED = "assistant_transcript_completed"

    ASSISTANT_STARTED_TOOL_CALL = "assistant_started_tool_call"
    ASSISTANT_RECEIVED_TOOL_CALL_RESULT = "assistant_received_tool_call"

    ASSISTANT_STARTED_MCP_TOOL_CALL = "assistant_started_mcp_tool_call"
    ASSISTANT_COMPLETED_MCP_TOOL_CALL_RESULT = "assistant_received_mcp_tool_call_result"
    ASSISTANT_FAILED_MCP_TOOL_CALL = "assistant_failed_mcp_tool_call"

    ASSISTANT_CONFIG_UPDATE_REQUEST = "assistant_config_update_request"

    TIMEOUT_OCCURRED = "timeout_occurred"
    IDLE_TRANSITION = "idle_transition"
    ERROR_OCCURRED = "error_occurred"

    def __str__(self) -> str:
        return self.value


class StateType(Enum):
    IDLE = "idle"
    TIMEOUT = "timeout"
    LISTENING = "listening"
    RESPONDING = "responding"
    TOOL_CALLING = "tool_calling"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


class AssistantState(ABC, LoggingMixin):
    def __init__(self, state_type: StateType):
        self._state_type = state_type

    @property
    def state_type(self) -> StateType:
        return self._state_type

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        pass

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        pass

    @abstractmethod
    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None: ...

    async def transition_to_idle(self, context: VoiceAssistantContext) -> None:
        from rtvoice.state.idle import IdleState

        await self._transition_to(IdleState(), context)

    async def _transition_to_timeout(self, context: VoiceAssistantContext) -> None:
        from rtvoice.state.timeout import TimeoutState

        await self._transition_to(TimeoutState(), context)

    async def _transition_to_listening(self, context: VoiceAssistantContext) -> None:
        from rtvoice.state.listening import ListeningState

        await self._transition_to(ListeningState(), context)

    async def _transition_to_responding(self, context: VoiceAssistantContext) -> None:
        from rtvoice.state.responding import RespondingState

        await self._transition_to(RespondingState(), context)

    async def _transition_to_tool_calling(self, context: VoiceAssistantContext) -> None:
        from rtvoice.state.tool_calling import ToolCallingState

        await self._transition_to(ToolCallingState(), context)

    async def _transition_to(
        self, new_state: AssistantState, context: VoiceAssistantContext
    ) -> None:
        self.logger.info(
            "Transitioning from %s to %s",
            self.__class__.__name__,
            new_state.__class__.__name__,
        )

        await self.on_exit(context)
        context.state = new_state

        self.logger.debug("Calling on_enter for %s", new_state.__class__.__name__)
        await context.state.on_enter(context)
        self.logger.debug("on_enter completed for %s", new_state.__class__.__name__)
