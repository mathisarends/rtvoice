from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from rtvoice.realtime.schemas import RealtimeSessionConfig

if TYPE_CHECKING:
    from rtvoice.watchdogs.conversation_history import ConversationTurn


# has to be send by agent
class AgentStartedEvent(BaseModel):
    session_config: RealtimeSessionConfig


class AgentStoppedEvent(BaseModel): ...


class ConversationHistoryResponseEvent(BaseModel):
    conversation_turns: list[ConversationTurn]


class SpeechSpeedUpdateRequestedEvent(BaseModel):
    speech_speed: float


class ConversationItemCreateRequestedEvent(BaseModel):
    content: str


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


class AssistantReceivedToolCallResultEvent(BaseModel):
    call_id: str
    result: Any


class ToolCallResultReadyEvent(BaseModel):
    call_id: str
    tool_name: str
    output: str
    response_instruction: str | None = None


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
