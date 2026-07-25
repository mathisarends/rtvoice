from typing import TYPE_CHECKING, Any

from transitbus import Event

if TYPE_CHECKING:
    from rtvoice.agent.views import AgentError
    from rtvoice.realtime.schemas import ToolChoiceMode

    type AgentErrorValue = AgentError
    type ToolChoiceValue = ToolChoiceMode
else:
    type AgentErrorValue = Any
    type ToolChoiceValue = Any


class UpdateSpeechSpeedCommand(Event):
    speed: float


class UpdateToolChoiceCommand(Event):
    tool_choice: ToolChoiceValue


class InterruptAssistantCommand(Event):
    pass


class StopAgentCommand(Event):
    pass


class AgentSessionConnectedEvent(Event):
    pass


class AgentStartingEvent(Event):
    pass


class AgentStoppedEvent(Event):
    pass


class UserTranscriptChunkReceivedEvent(Event):
    chunk: str


class UserTranscriptCompletedEvent(Event):
    transcript: str
    item_id: str


class AssistantTranscriptChunkReceivedEvent(Event):
    chunk: str


class AssistantTranscriptDeltaEvent(Event):
    delta: str
    item_id: str
    output_index: int
    content_index: int


class AssistantTranscriptCompletedEvent(Event):
    transcript: str
    item_id: str
    output_index: int
    content_index: int


class UserInactivityCountdownEvent(Event):
    remaining_seconds: int


class UserInactivityTimeoutEvent(Event):
    timeout_seconds: float


class AssistantInterruptedEvent(Event):
    item_id: str | None = None
    played_ms: int | None = None


class AudioPlaybackCompletedEvent(Event):
    pass


class AgentErrorEvent(Event):
    error: AgentErrorValue
    event_id: str | None = None


class UserStartedSpeakingEvent(Event):
    pass


class UserStoppedSpeakingEvent(Event):
    pass


class AssistantStartedRespondingEvent(Event):
    pass


class AssistantStoppedRespondingEvent(Event):
    pass
