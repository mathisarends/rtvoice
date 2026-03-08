from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from rtvoice.tools import RealtimeTools
    from rtvoice.views import (
        AgentError,
        AssistantVoice,
        NoiseReduction,
        RealtimeModel,
        TranscriptionModel,
        TurnDetection,
    )


@dataclass
class StartAgentCommand:
    model: RealtimeModel
    instructions: str
    voice: AssistantVoice
    speech_speed: float
    transcription_model: TranscriptionModel | None
    noise_reduction: NoiseReduction
    turn_detection: TurnDetection
    tools: RealtimeTools


@dataclass
class UpdateSpeechSpeedCommand:
    speed: float


class AgentSessionConnectedEvent(BaseModel):
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


class UserInactivityCountdownEvent(BaseModel):
    remaining_seconds: int


class UserInactivityTimeoutEvent(BaseModel):
    timeout_seconds: float


class AgentBusyEvent(BaseModel):
    busy: bool


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
