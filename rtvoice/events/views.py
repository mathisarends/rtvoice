from __future__ import annotations

from pydantic import BaseModel

from rtvoice.realtime.schemas import RealtimeSessionConfig


class AgentStartedEvent(BaseModel):
    session_config: RealtimeSessionConfig


class AgentStoppedEvent(BaseModel):
    pass


class StopAgentCommand(BaseModel):
    pass


class UserTranscriptChunkReceivedEvent(BaseModel):
    chunk: str


class UserTranscriptCompletedEvent(BaseModel):
    transcript: str
    item_id: str


class AssistantTranscriptChunkReceivedEvent(BaseModel):
    chunk: str


class AssistantTranscriptCompletedEvent(BaseModel):
    transcript: str
    item_id: str
    output_index: int
    content_index: int


class UserInactivityTimeoutEvent(BaseModel):
    timeout_seconds: float


class AssistantInterruptedEvent(BaseModel):
    pass


class AudioPlaybackCompletedEvent(BaseModel):
    pass
