from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from rtvoice.conversation.views import ConversationTurn


class RealtimeModel(StrEnum):
    """Available OpenAI Realtime API model variants.

    Attributes:
        GPT_REALTIME: Full-sized model with higher capability.
        GPT_REALTIME_MINI: Smaller, faster, and cheaper variant.
            Recommended default for most use-cases.
    """

    GPT_REALTIME = "gpt-realtime"
    GPT_REALTIME_MINI = "gpt-realtime-mini"
    GPT_REALTIME_1_5 = "gpt-realtime-1.5"


class AssistantVoice(StrEnum):
    """TTS voices available for the OpenAI Realtime API.

    Attributes:
        ALLOY: Neutral and balanced; clean output suitable for general use.
        ASH: Clear and precise; described as a male baritone with a slightly
            scratchy yet upbeat quality. May have limited performance with accents.
        BALLAD: Melodic and gentle; community notes suggest a male-sounding voice.
        CORAL: Warm and friendly; good for approachable or empathetic tones.
        ECHO: Resonant and deep; strong presence in delivery.
        FABLE: Narrative-like and expressive; fitting for storytelling contexts.
        ONYX: Darker, strong, and confident in tone.
        NOVA: Bright, youthful, and energetic.
        SAGE: Calm and thoughtful; measured pacing with a reflective quality.
        SHIMMER: Bright and energetic; dynamic expression with high clarity.
        VERSE: Versatile and expressive; adapts well across different contexts.
        CEDAR: Realtime-only voice. No official description available.
        MARIN: Realtime-only voice. No official description available.

    Example:
        ```python
        agent = RealtimeAgent(
            voice=AssistantVoice.CORAL,
        )
        ```
    """

    ALLOY = "alloy"
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SAGE = "sage"
    SHIMMER = "shimmer"
    VERSE = "verse"
    CEDAR = "cedar"
    MARIN = "marin"


class TranscriptionModel(StrEnum):
    """STT models used to produce user transcript events.

    Attributes:
        WHISPER_1: OpenAI Whisper v1. Currently the only supported model.

    Note:
        Pass `transcription_model=None` to `RealtimeAgent` to disable
        transcription entirely. Note that a supervisor agent requires
        transcription to be enabled.
    """

    WHISPER_1 = "whisper-1"


class NoiseReduction(StrEnum):
    """Microphone noise reduction profile applied to audio input.

    Attributes:
        NEAR_FIELD: Optimised for close-range audio, e.g. a headset microphone.
        FAR_FIELD: Optimised for distant audio sources, e.g. a desktop or room mic.

    Example:
        ```python
        agent = RealtimeAgent(
            noise_reduction=NoiseReduction.NEAR_FIELD,
        )
        ```
    """

    NEAR_FIELD = "near_field"
    FAR_FIELD = "far_field"


class SemanticEagerness(StrEnum):
    """Controls how quickly semantic VAD decides the user has finished speaking.

    Higher eagerness means the model cuts off sooner; lower eagerness waits
    longer to ensure the user has truly finished their thought.

    Attributes:
        LOW: Waits longest before committing to end-of-turn.
        MEDIUM: Balanced cut-off timing.
        HIGH: Cuts off quickly; may interrupt longer pauses mid-thought.
        AUTO: Let the model decide based on context. Recommended default.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class SemanticVAD(BaseModel):
    """Semantic voice-activity detection strategy.

    The model waits until it understands the speaker has finished a thought,
    producing more natural turn-taking with fewer false cut-offs compared to
    energy-based detection.

    Attributes:
        eagerness: How aggressively the model cuts off the user.
            Defaults to `SemanticEagerness.AUTO`.

    Example:
        ```python
        agent = RealtimeAgent(
            turn_detection=SemanticVAD(eagerness=SemanticEagerness.LOW),
        )
        ```
    """

    eagerness: SemanticEagerness = SemanticEagerness.AUTO
    """How quickly the model decides the user has stopped speaking."""


class ServerVAD(BaseModel):
    """Energy- and silence-based voice-activity detection strategy.

    Triggers end-of-turn based on audio energy thresholds and silence duration
    rather than semantic understanding. Useful when latency is critical or
    semantic VAD produces undesirable behaviour.

    Attributes:
        threshold: Energy threshold in the range `[0, 1]` above which audio
            is considered speech. Defaults to `0.5`.
        prefix_padding_ms: Milliseconds of audio to include before the detected
            speech onset. Defaults to `300`.
        silence_duration_ms: Milliseconds of silence required to commit an
            end-of-turn. Defaults to `500`.

    Example:
        ```python
        agent = RealtimeAgent(
            turn_detection=ServerVAD(silence_duration_ms=800),
        )
        ```
    """

    threshold: float = 0.5
    """Energy threshold above which audio is considered speech."""

    prefix_padding_ms: int = 300
    """Milliseconds of audio prepended before the detected speech onset."""

    silence_duration_ms: int = 500
    """Milliseconds of silence required to commit an end-of-turn."""


type TurnDetection = SemanticVAD | ServerVAD
"""Union type for voice-activity detection strategies.

Either a [`SemanticVAD`][rtvoice.views.SemanticVAD] or a
[`ServerVAD`][rtvoice.views.ServerVAD] instance. Passed directly to
`RealtimeAgent` via the `turn_detection` parameter.
"""


@dataclass
class AgentError:
    """Error information received in `AgentListener.on_agent_error`.

    This object is created internally and passed to your listener —
    you do not construct it yourself.

    Attributes:
        type: OpenAI error type identifier (e.g. `invalid_request_error`).
        message: Human-readable description of the error.

    Example:
        ```python
        class MyListener(AgentListener):
            async def on_agent_error(self, error: AgentError) -> None:
                if error.type == "invalid_request_error":
                    print(f"Bad request: {error.message}")
        ```
    """

    type: str
    """OpenAI error type identifier (e.g. `invalid_request_error`)."""

    message: str
    """Human-readable description of the error."""

    def __str__(self) -> str:
        return f"[{self.type}] {self.message}"


class AgentListener:
    """Callback interface for `RealtimeAgent` session lifecycle events.

    Subclass this and pass an instance to `RealtimeAgent` via the `listener`
    parameter to react to speaking state changes, transcripts, errors, and
    session boundaries.

    All methods are async no-ops by default — override only the ones you need.

    Example:
        ```python
        class MyListener(AgentListener):
            async def on_user_transcript(self, transcript: str) -> None:
                print(f"User: {transcript}")

            async def on_assistant_transcript(self, transcript: str) -> None:
                print(f"Assistant: {transcript}")


        agent = RealtimeAgent(listener=MyListener())
        ```
    """

    async def on_agent_starting(self) -> None:
        """Called immediately when run() is invoked, before any I/O or WebSocket setup.

        Use this to show loading states in the UI before the session is ready.
        """

    async def on_agent_session_connected(self) -> None:
        """Called once the WebSocket session has been established and is ready."""

    async def on_agent_stopped(self) -> None:
        """Called after the agent has fully shut down and `run()` is about to return."""

    async def on_user_inactivity_countdown(self, remaining_seconds: int) -> None:
        """Called each second during the countdown before the inactivity timeout fires.

        Fires at remaining_seconds = 5, 4, 3, 2, 1.

        Args:
            remaining_seconds: Seconds remaining until the session is stopped.
        """

    async def on_agent_interrupted(self) -> None:
        """Called when the assistant's response is interrupted by the user speaking."""

    async def on_agent_error(self, error: AgentError) -> None:
        """Called when the agent or the Realtime API encounters an error.

        Args:
            error: Structured error information including type, message, and
                optional code and parameter.
        """

    async def on_user_transcript(self, transcript: str) -> None:
        """Called when the user's speech has been fully transcribed.

        Only fires if a `transcription_model` is configured on the agent.

        Args:
            transcript: The finalised transcript text for the current user turn.
        """

    async def on_assistant_transcript(self, transcript: str) -> None:
        """Called when the assistant has finished generating a response and its
        transcript is complete.

        Args:
            transcript: The full transcript of the assistant's response.
        """

    async def on_user_started_speaking(self) -> None:
        """Called when VAD detects that the user has started speaking."""

    async def on_user_stopped_speaking(self) -> None:
        """Called when VAD detects that the user has stopped speaking."""

    async def on_assistant_started_responding(self) -> None:
        """Called when the assistant begins streaming an audio response."""

    async def on_assistant_stopped_responding(self) -> None:
        """Called when the assistant has finished streaming its audio response."""

    async def on_subagent_started(self) -> None:
        """Called when the supervisor agent starts running."""

    async def on_subagent_finished(self) -> None:
        """Called when the supervisor agent finishes running."""


class AgentResult(BaseModel):
    """Return value of `RealtimeAgent.run()` after the session ends.

    Attributes:
        turns: Ordered list of conversation turns recorded during the session.
        recording_path: Path to the recorded session audio file, or `None`
            if recording was not enabled.

    Example:
        ```python
        result = await agent.run()

        for turn in result.turns:
            print(turn)

        if result.recording_path:
            print(f"Recording saved to: {result.recording_path}")
        ```
    """

    turns: list[ConversationTurn]
    """Ordered list of conversation turns recorded during the session."""

    recording_path: Path | None = None
    """Path to the recorded session audio, or `None` if recording was disabled."""


@dataclass
class ClarificationCheckpoint:
    resume_history: list
    clarify_call_id: str
