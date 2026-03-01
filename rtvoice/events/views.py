from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from rtvoice.realtime.schemas import RealtimeSessionConfig


@dataclass
class StartAgentCommand:
    session_config: RealtimeSessionConfig


class AgentStartedEvent(BaseModel):
    pass


class AgentStoppedEvent(BaseModel):
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


class SubAgentCalledEvent(BaseModel):
    agent_name: str
    task: str


class AgentErrorEvent(BaseModel):
    type: str
    message: str
    code: str | None = None
    param: str | None = None
    event_id: str | None = None
