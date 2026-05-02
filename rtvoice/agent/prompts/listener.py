import logging

from rtvoice.agent.views import AgentError
from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentErrorEvent,
    AgentSessionConnectedEvent,
    AgentStartingEvent,
    AssistantInterruptedEvent,
    AssistantStartedRespondingEvent,
    AssistantStoppedRespondingEvent,
    AssistantTranscriptCompletedEvent,
    AssistantTranscriptDeltaEvent,
    SubAgentFinishedEvent,
    SubAgentStartedEvent,
    UserInactivityCountdownEvent,
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
    UserTranscriptCompletedEvent,
)

logger = logging.getLogger(__name__)


class AgentListener:
    """Callback interface for `RealtimeAgent` lifecycle events.

    Subclass and pass to `RealtimeAgent` via the `listener` parameter.
    All methods are async no-ops by default — override only what you need.

    Example:
    """

    async def on_agent_starting(self) -> None:
        """Before any I/O or WebSocket setup — use for loading states."""

    async def on_agent_session_connected(self) -> None:
        """WebSocket session established and ready."""

    async def on_agent_stopped(self) -> None:
        """Agent fully shut down, `run()` is about to return."""

    async def on_user_inactivity_countdown(self, remaining_seconds: int) -> None:
        """Fires each second (5→1) before the inactivity timeout triggers."""

    async def on_agent_interrupted(self) -> None:
        """Assistant response interrupted by the user speaking."""

    async def on_agent_error(self, error: AgentError) -> None:
        """Agent or Realtime API encountered an error."""

    async def on_user_transcript(self, transcript: str) -> None:
        """Finalised user transcript. Requires `transcription_model` to be set."""

    async def on_assistant_transcript(self, transcript: str) -> None:
        """Full transcript of the assistant's completed response."""

    async def on_assistant_transcript_delta(self, delta: str) -> None:
        """Incremental assistant text chunk as it streams."""

    async def on_user_started_speaking(self) -> None:
        """VAD detected user speech start."""

    async def on_user_stopped_speaking(self) -> None:
        """VAD detected user speech end."""

    async def on_assistant_started_responding(self) -> None:
        """Assistant began streaming an audio response."""

    async def on_assistant_stopped_responding(self) -> None:
        """Assistant finished streaming its audio response."""

    async def on_subagent_started(self, agent_name: str) -> None:
        """Subagent `agent_name` started running."""

    async def on_subagent_finished(self, agent_name: str) -> None:
        """Subagent `agent_name` finished running."""


class AgentListenerBridge:
    def __init__(
        self,
        *,
        event_bus: EventBus,
        listener: AgentListener,
        inactivity_timeout_enabled: bool,
        has_subagents: bool,
        assistant_text_enabled: bool,
    ) -> None:
        self._event_bus = event_bus
        self._listener = listener
        self._inactivity_timeout_enabled = inactivity_timeout_enabled
        self._has_subagents = has_subagents
        self._assistant_text_enabled = assistant_text_enabled

    def setup(self) -> None:
        self._warn_countdown_mismatch_if_necessary()
        self._warn_subagent_mismatch_if_necessary()
        self._warn_text_modality_mismatch_if_necessary()

        self._event_bus.subscribe(
            UserTranscriptCompletedEvent, self._on_user_transcript_completed
        )
        self._event_bus.subscribe(
            AssistantTranscriptCompletedEvent,
            self._on_assistant_transcript_completed,
        )
        self._event_bus.subscribe(
            AssistantTranscriptDeltaEvent,
            self._on_assistant_transcript_delta,
        )
        self._event_bus.subscribe(AgentStartingEvent, self._on_agent_starting)
        self._event_bus.subscribe(
            AgentSessionConnectedEvent,
            self._on_agent_session_connected,
        )
        self._event_bus.subscribe(
            AssistantInterruptedEvent,
            self._on_assistant_interrupted,
        )
        self._event_bus.subscribe(AgentErrorEvent, self._on_agent_error)
        self._event_bus.subscribe(
            UserStartedSpeakingEvent,
            self._on_user_started_speaking,
        )
        self._event_bus.subscribe(
            UserStoppedSpeakingEvent,
            self._on_user_stopped_speaking,
        )
        self._event_bus.subscribe(
            AssistantStartedRespondingEvent,
            self._on_assistant_started_responding,
        )
        self._event_bus.subscribe(
            AssistantStoppedRespondingEvent,
            self._on_assistant_stopped_responding,
        )
        self._event_bus.subscribe(
            UserInactivityCountdownEvent,
            self._on_user_inactivity_countdown,
        )
        self._event_bus.subscribe(SubAgentStartedEvent, self._on_subagent_started)
        self._event_bus.subscribe(SubAgentFinishedEvent, self._on_subagent_finished)

    async def _on_user_transcript_completed(
        self, event: UserTranscriptCompletedEvent
    ) -> None:
        await self._listener.on_user_transcript(event.transcript)

    async def _on_assistant_transcript_completed(
        self, event: AssistantTranscriptCompletedEvent
    ) -> None:
        await self._listener.on_assistant_transcript(event.transcript)

    async def _on_assistant_transcript_delta(
        self, event: AssistantTranscriptDeltaEvent
    ) -> None:
        await self._listener.on_assistant_transcript_delta(event.delta)

    async def _on_agent_starting(self, _: AgentStartingEvent) -> None:
        await self._listener.on_agent_starting()

    async def _on_agent_session_connected(self, _: AgentSessionConnectedEvent) -> None:
        await self._listener.on_agent_session_connected()

    async def _on_assistant_interrupted(self, _: AssistantInterruptedEvent) -> None:
        await self._listener.on_agent_interrupted()

    async def _on_agent_error(self, event: AgentErrorEvent) -> None:
        await self._listener.on_agent_error(event.error)

    async def _on_user_started_speaking(self, _: UserStartedSpeakingEvent) -> None:
        await self._listener.on_user_started_speaking()

    async def _on_user_stopped_speaking(self, _: UserStoppedSpeakingEvent) -> None:
        await self._listener.on_user_stopped_speaking()

    async def _on_assistant_started_responding(
        self, _: AssistantStartedRespondingEvent
    ) -> None:
        await self._listener.on_assistant_started_responding()

    async def _on_assistant_stopped_responding(
        self, _: AssistantStoppedRespondingEvent
    ) -> None:
        await self._listener.on_assistant_stopped_responding()

    async def _on_user_inactivity_countdown(
        self, event: UserInactivityCountdownEvent
    ) -> None:
        await self._listener.on_user_inactivity_countdown(event.remaining_seconds)

    async def _on_subagent_started(self, event: SubAgentStartedEvent) -> None:
        await self._listener.on_subagent_started(event.agent_name)

    async def _on_subagent_finished(self, event: SubAgentFinishedEvent) -> None:
        await self._listener.on_subagent_finished(event.agent_name)

    def _warn_countdown_mismatch_if_necessary(self) -> None:
        overrides_countdown = self._listener_overrides_countdown()
        listener_name = type(self._listener).__name__

        if overrides_countdown and not self._inactivity_timeout_enabled:
            logger.warning(
                "Listener '%s' overrides on_user_inactivity_countdown "
                "but inactivity_timeout_enabled is False - callback will never fire.",
                listener_name,
            )

        if self._inactivity_timeout_enabled and not overrides_countdown:
            logger.warning(
                "inactivity_timeout_enabled is True but listener '%s' does not override "
                "on_user_inactivity_countdown - countdown events will be silently ignored.",
                listener_name,
            )

    def _listener_overrides_countdown(self) -> bool:
        cls = type(self._listener)
        listener_method = getattr(cls, "on_user_inactivity_countdown", None)
        if listener_method is None:
            return False
        return listener_method is not AgentListener.on_user_inactivity_countdown

    def _warn_subagent_mismatch_if_necessary(self) -> None:
        if self._listener_overrides_subagent_callbacks() and not self._has_subagents:
            logger.warning(
                "Listener '%s' overrides on_subagent_started or on_subagent_finished "
                "but no subagents are configured - callbacks will never fire.",
                type(self._listener).__name__,
            )

    def _warn_text_modality_mismatch_if_necessary(self) -> None:
        if (
            self._listener_overrides_assistant_transcript_delta()
            and not self._assistant_text_enabled
        ):
            logger.warning(
                "Listener '%s' overrides on_assistant_transcript_delta "
                "but output_modalities does not include 'text' - callback will never fire.",
                type(self._listener).__name__,
            )

    def _listener_overrides_assistant_transcript_delta(self) -> bool:
        cls = type(self._listener)
        delta = getattr(cls, "on_assistant_transcript_delta", None)
        return (
            delta is not None
            and delta is not AgentListener.on_assistant_transcript_delta
        )

    def _listener_overrides_subagent_callbacks(self) -> bool:
        cls = type(self._listener)
        started = getattr(cls, "on_subagent_started", None)
        finished = getattr(cls, "on_subagent_finished", None)
        return (
            started is not None and started is not AgentListener.on_subagent_started
        ) or (
            finished is not None and finished is not AgentListener.on_subagent_finished
        )
