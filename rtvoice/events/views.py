from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from rtvoice.realtime.schemas import FunctionTool, ToolChoiceMode
    from rtvoice.tools import Tools
    from rtvoice.views import (
        AgentError,
        AssistantVoice,
        NoiseReduction,
        OutputModality,
        RealtimeModel,
        TranscriptionModel,
        TurnDetection,
    )


@dataclass
class StartAgentCommand:
    model: RealtimeModel
    voice: AssistantVoice
    speech_speed: float
    transcription_model: TranscriptionModel | None
    output_modalities: list[OutputModality]
    noise_reduction: NoiseReduction
    turn_detection: TurnDetection
    tools: Tools
    instructions: str = ""


@dataclass
class ConfigureSessionCommand:
    model: RealtimeModel
    voice: AssistantVoice
    speech_speed: float
    transcription_model: TranscriptionModel | None
    output_modalities: list[OutputModality]
    noise_reduction: NoiseReduction
    turn_detection: TurnDetection
    tools: Tools
    instructions: str = ""


@dataclass
class UpdateSpeechSpeedCommand:
    speed: float


@dataclass
class UpdateToolChoiceCommand:
    tool_choice: ToolChoiceMode


@dataclass
class CancelSubAgentCommand:
    pass


@dataclass
class UpdateSessionToolsCommand:
    tools: list[FunctionTool]


class AgentSessionConnectedEvent(BaseModel):
    pass


class AgentStartingEvent(BaseModel):
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


class AssistantTranscriptDeltaEvent(BaseModel):
    delta: str
    item_id: str
    output_index: int
    content_index: int


class AssistantTranscriptCompletedEvent(BaseModel):
    transcript: str
    item_id: str
    output_index: int
    content_index: int


class UserInactivityCountdownEvent(BaseModel):
    remaining_seconds: int


class UserInactivityTimeoutEvent(BaseModel):
    timeout_seconds: float


class SubAgentStartedEvent(BaseModel):
    agent_name: str


class SubAgentFinishedEvent(BaseModel):
    agent_name: str


class AssistantInterruptedEvent(BaseModel):
    pass


class AudioPlaybackCompletedEvent(BaseModel):
    pass


class AgentErrorEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    error: AgentError
    event_id: str | None = None


class UserStartedSpeakingEvent(BaseModel):
    pass


class UserStoppedSpeakingEvent(BaseModel):
    pass


class AssistantStartedRespondingEvent(BaseModel):
    pass


class AssistantStoppedRespondingEvent(BaseModel):
    pass
