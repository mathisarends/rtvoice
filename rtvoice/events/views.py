from typing import Any

from pydantic import BaseModel

from rtvoice.realtime.schemas import RealtimeSessionConfig


# has to be send by agent
class AgentStartedEvent(BaseModel):
    session_config: RealtimeSessionConfig


class AgentStoppedEvent(BaseModel): ...


# TODO: AgentSpeechSpeedChanged

# TODO: AgentSpeecInterrupted


class UserStartedSpeakingEvent(BaseModel):
    pass


class UserSpeechEndedEvent(BaseModel):
    pass


class UserTranscriptChunkReceivedEvent(BaseModel):
    chunk: str


class UserTranscriptCompletedEvent(BaseModel):
    transcript: str
    item_id: str


class AudioChunkReceivedEvent(BaseModel):
    audio_data: bytes
    response_id: str
    item_id: str
    output_index: int
    content_index: int


class AssistantStartedRespondingEvent(BaseModel):
    pass


class AssistantResponseCompletedEvent(BaseModel):
    pass


class AssistantSpeechInterruptedEvent(BaseModel):
    item_id: str
    audio_end_ms: int


class MessageTruncationRequestedEvent(BaseModel):
    item_id: str
    audio_end_ms: int


class AssistantStartedResponseEvent(BaseModel):
    response_id: str


class AssistantCompletedResponseEvent(BaseModel):
    response_id: str


class AssistantTranscriptChunkReceivedEvent(BaseModel):
    chunk: str


class AssistantTranscriptCompletedEvent(BaseModel):
    transcript: str
    item_id: str
    output_index: int
    content_index: int


class AssistantStartedToolCallEvent(BaseModel):
    call_id: str
    name: str
    arguments: str


class AssistantReceivedToolCallResultEvent(BaseModel):
    call_id: str
    result: Any


class AssistantStartedMCPToolCallEvent(BaseModel):
    pass


class AssistantCompletedMCPToolCallResultEvent(BaseModel):
    result: Any


class AssistantFailedMCPToolCallEvent(BaseModel):
    error_type: str
    error_message: str
    error_details: dict[str, Any]


class AssistantConfigUpdateRequestEvent(BaseModel):
    config: dict[str, Any]


class TimeoutOccurredEvent(BaseModel):
    timeout_seconds: float


class IdleTransitionEvent(BaseModel):
    pass


class ErrorOccurredEvent(BaseModel):
    error_type: str
    error_code: str | None = None
    error_message: str
    details: dict[str, Any] | None = None


# ============================================================================
# Legacy Enum (for backward compatibility)
# ============================================================================

from enum import StrEnum


class VoiceAssistantEvent(StrEnum):
    WAKE_WORD_DETECTED = "wake_word_detected"
    USER_STARTED_SPEAKING = "user_started_speaking"
    USER_SPEECH_ENDED = "user_speech_ended"
    USER_TRANSCRIPT_CHUNK_RECEIVED = "user_transcript_chunk_received"
    USER_TRANSCRIPT_COMPLETED = "user_transcript_completed"
    AUDIO_CHUNK_RECEIVED = "audio_chunk_received"
    ASSISTANT_STARTED_RESPONDING = "assistant_started_responding"
    ASSISTANT_RESPONSE_COMPLETED = "assistant_response_completed"
    ASSISTANT_SPEECH_INTERRUPTED = "assistant_speech_interrupted"
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
