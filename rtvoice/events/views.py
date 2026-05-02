from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtvoice.agent.views import (
        AgentError,
    )
    from rtvoice.realtime.schemas import FunctionTool, ToolChoiceMode


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


@dataclass
class AgentSessionConnectedEvent:
    pass


@dataclass
class AgentStartingEvent:
    pass


@dataclass
class AgentStoppedEvent:
    pass


@dataclass
class UserTranscriptChunkReceivedEvent:
    chunk: str


@dataclass
class UserTranscriptCompletedEvent:
    transcript: str
    item_id: str


@dataclass
class AssistantTranscriptChunkReceivedEvent:
    chunk: str


@dataclass
class AssistantTranscriptDeltaEvent:
    delta: str
    item_id: str
    output_index: int
    content_index: int


@dataclass
class AssistantTranscriptCompletedEvent:
    transcript: str
    item_id: str
    output_index: int
    content_index: int


@dataclass
class UserInactivityCountdownEvent:
    remaining_seconds: int


@dataclass
class UserInactivityTimeoutEvent:
    timeout_seconds: float


@dataclass
class SubAgentStartedEvent:
    agent_name: str


@dataclass
class SubAgentFinishedEvent:
    agent_name: str


@dataclass
class AssistantInterruptedEvent:
    item_id: str | None = None
    played_ms: int | None = None


@dataclass
class AudioPlaybackCompletedEvent:
    pass


@dataclass
class AgentErrorEvent:
    error: AgentError
    event_id: str | None = None


@dataclass
class UserStartedSpeakingEvent:
    pass


@dataclass
class UserStoppedSpeakingEvent:
    pass


@dataclass
class AssistantStartedRespondingEvent:
    pass


@dataclass
class AssistantStoppedRespondingEvent:
    pass
