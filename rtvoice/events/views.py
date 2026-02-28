from pydantic import BaseModel

from rtvoice.realtime.schemas import RealtimeSessionConfig


class StartAgentCommand(BaseModel):
    session_config: RealtimeSessionConfig


class AgentStartedEvent(BaseModel):
    pass


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


class SubAgentCalledEvent(BaseModel):
    agent_name: str
    task: str


class AgentErrorEvent(BaseModel):
    type: str
    message: str
    code: str | None = None
    param: str | None = None
    event_id: str | None = None
