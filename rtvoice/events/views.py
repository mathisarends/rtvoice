from typing import TYPE_CHECKING, Any

from transitbus import Event

from rtvoice.shared.speech_speed import DEFAULT_SPEECH_SPEED, SpeechSpeed

if TYPE_CHECKING:
    from rtvoice.agent.views import AgentError
    from rtvoice.realtime.schemas import ToolChoiceMode
    from rtvoice.tools.views import ActionKind

    type ActionKindValue = ActionKind
    type AgentErrorValue = AgentError
    type ToolChoiceValue = ToolChoiceMode
else:
    type ActionKindValue = Any
    type AgentErrorValue = Any
    type ToolChoiceValue = Any


class UpdateSpeechSpeedCommand(Event):
    # clamped on construction, so no dispatcher can emit an out-of-range speed
    speed: SpeechSpeed


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
    response_id: str | None = None


class UserInactivityCountdownEvent(Event):
    remaining_seconds: int


class UserInactivityTimeoutEvent(Event):
    timeout_seconds: float


class ToolExecutionStartedEvent(Event):
    response_id: str


class ToolExecutionCompletedEvent(Event):
    response_id: str
    response_pending: bool


class ToolExecutedEvent(Event):
    name: str
    action_kind: ActionKindValue
    silent: bool
    result: str


class AssistantInterruptedEvent(Event):
    item_id: str | None = None
    played_ms: int | None = None
    response_id: str | None = None
    speech_speed: SpeechSpeed = DEFAULT_SPEECH_SPEED


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
