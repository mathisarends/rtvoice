import logging

import pytest

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
from rtvoice.listener import AgentListener, AgentListenerBridge
from rtvoice.views import AgentError

AgentErrorEvent.model_rebuild(_types_namespace={"AgentError": AgentError})


def make_bridge(
    listener: AgentListener,
    *,
    inactivity_timeout_enabled: bool = False,
    has_subagents: bool = False,
    assistant_text_enabled: bool = False,
) -> AgentListenerBridge:
    return AgentListenerBridge(
        event_bus=EventBus(),
        listener=listener,
        inactivity_timeout_enabled=inactivity_timeout_enabled,
        has_subagents=has_subagents,
        assistant_text_enabled=assistant_text_enabled,
    )


class ListenerWithoutOverrides(AgentListener):
    pass


class ListenerWithCountdown(AgentListener):
    async def on_user_inactivity_countdown(self, remaining_seconds: int) -> None:
        _ = remaining_seconds


class ListenerWithTranscriptDelta(AgentListener):
    async def on_assistant_transcript_delta(self, delta: str) -> None:
        _ = delta


class ListenerWithSubagentStarted(AgentListener):
    async def on_subagent_started(self, agent_name: str) -> None:
        _ = agent_name


class ListenerWithSubagentFinished(AgentListener):
    async def on_subagent_finished(self, agent_name: str) -> None:
        _ = agent_name


class RecordingListener(AgentListener):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | int | AgentError | None]] = []

    async def on_agent_starting(self) -> None:
        self.calls.append(("on_agent_starting", None))

    async def on_agent_session_connected(self) -> None:
        self.calls.append(("on_agent_session_connected", None))

    async def on_agent_interrupted(self) -> None:
        self.calls.append(("on_agent_interrupted", None))

    async def on_agent_error(self, error: AgentError) -> None:
        self.calls.append(("on_agent_error", error))

    async def on_user_transcript(self, transcript: str) -> None:
        self.calls.append(("on_user_transcript", transcript))

    async def on_assistant_transcript(self, transcript: str) -> None:
        self.calls.append(("on_assistant_transcript", transcript))

    async def on_assistant_transcript_delta(self, delta: str) -> None:
        self.calls.append(("on_assistant_transcript_delta", delta))

    async def on_user_started_speaking(self) -> None:
        self.calls.append(("on_user_started_speaking", None))

    async def on_user_stopped_speaking(self) -> None:
        self.calls.append(("on_user_stopped_speaking", None))

    async def on_assistant_started_responding(self) -> None:
        self.calls.append(("on_assistant_started_responding", None))

    async def on_assistant_stopped_responding(self) -> None:
        self.calls.append(("on_assistant_stopped_responding", None))

    async def on_user_inactivity_countdown(self, remaining_seconds: int) -> None:
        self.calls.append(("on_user_inactivity_countdown", remaining_seconds))

    async def on_subagent_started(self, agent_name: str) -> None:
        self.calls.append(("on_subagent_started", agent_name))

    async def on_subagent_finished(self, agent_name: str) -> None:
        self.calls.append(("on_subagent_finished", agent_name))


class TestListenerOverrideChecks:
    def test_listener_overrides_countdown_is_false_by_default(self) -> None:
        bridge = make_bridge(ListenerWithoutOverrides())
        assert bridge._listener_overrides_countdown() is False

    def test_listener_overrides_countdown_is_true_when_implemented(self) -> None:
        bridge = make_bridge(ListenerWithCountdown())
        assert bridge._listener_overrides_countdown() is True

    def test_listener_overrides_assistant_delta_is_false_by_default(self) -> None:
        bridge = make_bridge(ListenerWithoutOverrides())
        assert bridge._listener_overrides_assistant_transcript_delta() is False

    def test_listener_overrides_assistant_delta_is_true_when_implemented(self) -> None:
        bridge = make_bridge(ListenerWithTranscriptDelta())
        assert bridge._listener_overrides_assistant_transcript_delta() is True

    def test_listener_overrides_subagent_callbacks_is_false_by_default(self) -> None:
        bridge = make_bridge(ListenerWithoutOverrides())
        assert bridge._listener_overrides_subagent_callbacks() is False

    def test_listener_overrides_subagent_callbacks_is_true_when_started_is_implemented(
        self,
    ) -> None:
        bridge = make_bridge(ListenerWithSubagentStarted())
        assert bridge._listener_overrides_subagent_callbacks() is True

    def test_listener_overrides_subagent_callbacks_is_true_when_finished_is_implemented(
        self,
    ) -> None:
        bridge = make_bridge(ListenerWithSubagentFinished())
        assert bridge._listener_overrides_subagent_callbacks() is True


class TestListenerBridgeWarnings:
    def test_warns_when_subagent_callbacks_are_overridden_without_subagents(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_bridge(
                ListenerWithSubagentStarted(),
                has_subagents=False,
            ).setup()

        assert any(
            "on_subagent_started or on_subagent_finished" in r.message
            for r in caplog.records
        )

    def test_does_not_warn_when_subagent_callbacks_are_overridden_with_subagents(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_bridge(
                ListenerWithSubagentFinished(),
                has_subagents=True,
            ).setup()

        assert not any(
            "on_subagent_started or on_subagent_finished" in r.message
            for r in caplog.records
        )

    def test_warns_when_delta_callback_is_overridden_without_text_modality(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_bridge(
                ListenerWithTranscriptDelta(),
                assistant_text_enabled=False,
            ).setup()

        assert any("on_assistant_transcript_delta" in r.message for r in caplog.records)

    def test_does_not_warn_when_delta_callback_is_overridden_with_text_modality(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_bridge(
                ListenerWithTranscriptDelta(),
                assistant_text_enabled=True,
            ).setup()

        assert not any(
            "on_assistant_transcript_delta" in r.message for r in caplog.records
        )


class TestListenerBridgeEventPropagation:
    @pytest.mark.asyncio
    async def test_propagates_all_supported_events_to_listener(self) -> None:
        listener = RecordingListener()
        bridge = make_bridge(
            listener,
            inactivity_timeout_enabled=True,
            has_subagents=True,
            assistant_text_enabled=True,
        )
        bridge.setup()

        await bridge._event_bus.dispatch(AgentStartingEvent())
        await bridge._event_bus.dispatch(AgentSessionConnectedEvent())
        await bridge._event_bus.dispatch(
            UserTranscriptCompletedEvent(transcript="hello", item_id="item-1")
        )
        await bridge._event_bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="hi there",
                item_id="item-2",
                output_index=0,
                content_index=0,
            )
        )
        await bridge._event_bus.dispatch(
            AssistantTranscriptDeltaEvent(
                delta="hi",
                item_id="item-2",
                output_index=0,
                content_index=0,
            )
        )
        await bridge._event_bus.dispatch(AssistantInterruptedEvent())
        error = AgentError(type="internal_error", message="broken")
        await bridge._event_bus.dispatch(AgentErrorEvent(error=error))
        await bridge._event_bus.dispatch(UserStartedSpeakingEvent())
        await bridge._event_bus.dispatch(UserStoppedSpeakingEvent())
        await bridge._event_bus.dispatch(AssistantStartedRespondingEvent())
        await bridge._event_bus.dispatch(AssistantStoppedRespondingEvent())
        await bridge._event_bus.dispatch(
            UserInactivityCountdownEvent(remaining_seconds=5)
        )
        await bridge._event_bus.dispatch(SubAgentStartedEvent(agent_name="research"))
        await bridge._event_bus.dispatch(SubAgentFinishedEvent(agent_name="research"))

        assert listener.calls == [
            ("on_agent_starting", None),
            ("on_agent_session_connected", None),
            ("on_user_transcript", "hello"),
            ("on_assistant_transcript", "hi there"),
            ("on_assistant_transcript_delta", "hi"),
            ("on_agent_interrupted", None),
            ("on_agent_error", error),
            ("on_user_started_speaking", None),
            ("on_user_stopped_speaking", None),
            ("on_assistant_started_responding", None),
            ("on_assistant_stopped_responding", None),
            ("on_user_inactivity_countdown", 5),
            ("on_subagent_started", "research"),
            ("on_subagent_finished", "research"),
        ]
