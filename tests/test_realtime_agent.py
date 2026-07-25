import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rtvoice.agent import RealtimeAgent, Subagent
from rtvoice.agent.listener import AgentListener
from rtvoice.agent.views import (
    AgentError,
    AssistantVoice,
    InjectedConversation,
    InjectedUserMessage,
    NoiseReduction,
    RealtimeModel,
    ReasoningEffort,
    SemanticVAD,
    ServerVAD,
    TranscriptionModel,
)
from rtvoice.events.views import (
    AgentErrorEvent,
    AgentSessionConnectedEvent,
    AgentStoppedEvent,
    AssistantInterruptedEvent,
    AssistantStartedRespondingEvent,
    AssistantStoppedRespondingEvent,
    AssistantTranscriptCompletedEvent,
    AssistantTranscriptDeltaEvent,
    AudioPlaybackCompletedEvent,
    InterruptAssistantCommand,
    StopAgentCommand,
    UserInactivityTimeoutEvent,
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
    UserTranscriptCompletedEvent,
)


def make_agent(**kwargs) -> RealtimeAgent:
    audio_input = MagicMock()
    audio_output = MagicMock()
    with patch("rtvoice.agent.realtime_agent.OpenAIProvider"):
        return RealtimeAgent(
            audio_input=audio_input,
            audio_output=audio_output,
            **kwargs,
        )


class TestInitDefaults:
    def test_default_model_is_realtime_2_1_mini(self) -> None:
        agent = make_agent()
        assert agent._realtime_session._model == RealtimeModel.GPT_REALTIME_2_1_MINI

    def test_default_reasoning_effort_is_low(self) -> None:
        agent = make_agent()
        assert agent._realtime_session._reasoning_effort == ReasoningEffort.LOW

    def test_default_voice_is_marin(self) -> None:
        agent = make_agent()
        assert agent._realtime_session._voice == AssistantVoice.MARIN

    def test_default_noise_reduction_is_far_field(self) -> None:
        agent = make_agent()
        assert agent._realtime_session._noise_reduction == NoiseReduction.FAR_FIELD

    def test_default_turn_detection_is_semantic_vad(self) -> None:
        agent = make_agent()
        assert isinstance(agent._realtime_session._turn_detection, SemanticVAD)

    def test_default_transcription_model_is_whisper(self) -> None:
        agent = make_agent()
        assert (
            agent._realtime_session._transcription_model == TranscriptionModel.WHISPER_1
        )

    def test_default_inactivity_timeout_disabled(self) -> None:
        agent = make_agent()
        assert agent._realtime_session._inactivity_timeout_seconds is None

    @pytest.mark.asyncio
    async def test_subagent_is_executed_as_injected_regular_tool(self) -> None:
        subagent = Subagent(
            description="A subagent",
            instructions="Do the task.",
            llm=MagicMock(),
        )
        subagent.start = AsyncMock(return_value="final result")
        agent = make_agent(subagent=subagent)

        result = await agent._tools.execute("subagent", {"task": "Plan my day"})

        assert result.value == "final result"
        subagent.start.assert_awaited_once_with(
            "Plan my day", context="(no conversation yet)"
        )

    def test_default_injected_conversation_is_none(self) -> None:
        agent = make_agent()
        assert agent._realtime_session._injected_conversation is None

    def test_injected_conversation_is_passed_to_realtime_session(self) -> None:
        conversation = InjectedConversation([InjectedUserMessage("Mein Name ist Max.")])
        agent = make_agent(injected_conversation=conversation)
        assert agent._realtime_session._injected_conversation is conversation

    def test_custom_turn_detection_is_stored(self) -> None:
        vad = ServerVAD(silence_duration_ms=800)
        agent = make_agent(turn_detection=vad)
        assert agent._realtime_session._turn_detection == vad

    def test_custom_model_is_stored(self) -> None:
        agent = make_agent(model=RealtimeModel.GPT_REALTIME_2)
        assert agent._realtime_session._model == RealtimeModel.GPT_REALTIME_2

    @pytest.mark.parametrize(
        "model",
        [RealtimeModel.GPT_REALTIME, RealtimeModel.GPT_REALTIME_MINI],
    )
    def test_deprecated_model_emits_warning(self, model: RealtimeModel) -> None:
        with pytest.warns(DeprecationWarning, match="2027-01-20"):
            make_agent(model=model)

    def test_custom_reasoning_effort_is_stored(self) -> None:
        agent = make_agent(reasoning_effort=ReasoningEffort.MINIMAL)
        assert agent._realtime_session._reasoning_effort == ReasoningEffort.MINIMAL

    def test_recording_path_is_converted_to_path_object(self, tmp_path) -> None:
        agent = make_agent(recording_path=str(tmp_path / "rec.wav"))
        from pathlib import Path

        assert agent._realtime_session._recording_path == Path(tmp_path / "rec.wav")

    def test_recording_path_none_when_not_provided(self) -> None:
        agent = make_agent()
        assert agent._realtime_session._recording_path is None

    def test_stop_not_called_initially(self) -> None:
        agent = make_agent()
        assert agent._stop_called is False

    def test_text_output_mode_does_not_require_transcription_watchdog_field(
        self,
    ) -> None:
        agent = make_agent(transcription_model=None, output_modalities=["text"])
        assert not hasattr(agent, "_transcription_watchdog")


class TestSpeechSpeedClipping:
    def test_value_within_range_is_unchanged(self) -> None:
        agent = make_agent(speech_speed=1.0)
        assert agent._realtime_session._speech_speed == 1.0

    def test_speech_speed_below_minimum_is_clipped_to_minimum(self) -> None:
        agent = make_agent(speech_speed=0.1)
        assert agent._realtime_session._speech_speed == 0.25

    def test_value_above_maximum_is_clipped_to_one_point_five(self) -> None:
        agent = make_agent(speech_speed=2.0)
        assert agent._realtime_session._speech_speed == 1.5

    def test_exact_minimum_is_not_clipped(self) -> None:
        agent = make_agent(speech_speed=0.5)
        assert agent._realtime_session._speech_speed == 0.5

    def test_exact_maximum_is_not_clipped(self) -> None:
        agent = make_agent(speech_speed=1.5)
        assert agent._realtime_session._speech_speed == 1.5

    def test_out_of_range_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(speech_speed=3.0)
        assert any("out of range" in r.message for r in caplog.records)

    def test_in_range_does_not_log_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(speech_speed=1.2)
        assert not any("out of range" in r.message for r in caplog.records)


class TestInitWarnings:
    def test_no_warning_when_inactivity_timeout_is_unset(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent()
        assert not any(
            "inactivity_timeout_seconds" in r.message for r in caplog.records
        )


class TestInactivityTimeout:
    def test_seconds_enable_the_monitor(self) -> None:
        agent = make_agent(inactivity_timeout_seconds=30.0)
        assert agent._realtime_session._inactivity_timeout_seconds == 30.0
        assert hasattr(agent._realtime_session, "_conversation_inactivity_monitor")

    def test_no_monitor_without_seconds(self) -> None:
        agent = make_agent()
        assert not hasattr(agent._realtime_session, "_conversation_inactivity_monitor")


class TestStop:
    @pytest.mark.asyncio
    async def test_dispatches_agent_stopped_event(self) -> None:
        agent = make_agent()
        received = []

        async def capture(e: AgentStoppedEvent) -> None:
            received.append(e)

        agent._event_bus.on(AgentStoppedEvent, capture)

        await agent.stop()

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_sets_stop_called_flag(self) -> None:
        agent = make_agent()
        await agent.stop()
        assert agent._stop_called is True

    @pytest.mark.asyncio
    async def test_sets_stopped_event(self) -> None:
        agent = make_agent()
        await agent.stop()
        assert agent._stopped.is_set()

    @pytest.mark.asyncio
    async def test_is_idempotent(self) -> None:
        agent = make_agent()
        dispatched = []

        async def capture(e: AgentStoppedEvent) -> None:
            dispatched.append(e)

        agent._event_bus.on(AgentStoppedEvent, capture)

        await agent.stop()
        await agent.stop()

        assert len(dispatched) == 1

    @pytest.mark.asyncio
    async def test_calls_listener_on_agent_stopped(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)

        await agent.stop()

        listener.on_agent_stopped.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_listener_stop_does_not_raise(self) -> None:
        agent = make_agent()
        await agent.stop()


class TestStopTool:
    def test_stop_tool_is_registered_by_default(self) -> None:
        agent = make_agent()
        assert agent._tools.get("stop") is not None

    def test_stop_tool_is_exposed_in_schema(self) -> None:
        agent = make_agent()
        assert "stop" in [tool.name for tool in agent._tools.get_schema()]

    @pytest.mark.asyncio
    async def test_stop_tool_dispatches_stop_agent_command(self) -> None:
        agent = make_agent()
        received = []

        async def capture(e: StopAgentCommand) -> None:
            received.append(e)

        agent._event_bus.on(StopAgentCommand, capture)

        result = await agent._tools.execute("stop")

        assert result.ok
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_stop_tool_does_not_stop_before_playback_completed(self) -> None:
        agent = make_agent()

        await agent._tools.execute("stop")

        assert agent._stop_requested is True
        assert agent._stop_called is False

    @pytest.mark.asyncio
    async def test_agent_stops_once_playback_completed(self) -> None:
        agent = make_agent()

        await agent._tools.execute("stop")
        await agent._event_bus.dispatch(AudioPlaybackCompletedEvent())

        assert agent._stop_called is True

    @pytest.mark.asyncio
    async def test_playback_completed_without_stop_request_keeps_agent_running(
        self,
    ) -> None:
        agent = make_agent()

        await agent._event_bus.dispatch(AudioPlaybackCompletedEvent())

        assert agent._stop_called is False


class TestInterrupt:
    @pytest.mark.asyncio
    async def test_interrupt_dispatches_interrupt_command(self) -> None:
        agent = make_agent()
        received = []

        async def capture(e: InterruptAssistantCommand) -> None:
            received.append(e)

        agent._event_bus.on(InterruptAssistantCommand, capture)

        await agent.interrupt()

        assert len(received) == 1


class TestListenerWiring:
    @pytest.mark.asyncio
    async def test_on_user_transcript_is_called(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)

        await agent._event_bus.dispatch(
            UserTranscriptCompletedEvent(transcript="hello", item_id="x")
        )

        listener.on_user_transcript.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_on_assistant_transcript_is_called(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)

        await agent._event_bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="hi there",
                item_id="y",
                output_index=0,
                content_index=0,
            )
        )

        listener.on_assistant_transcript.assert_called_once_with("hi there")

    @pytest.mark.asyncio
    async def test_on_assistant_transcript_delta_is_called(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)

        await agent._event_bus.dispatch(
            AssistantTranscriptDeltaEvent(
                delta="hi",
                item_id="y",
                output_index=0,
                content_index=0,
            )
        )

        listener.on_assistant_transcript_delta.assert_called_once_with("hi")

    @pytest.mark.asyncio
    async def test_on_agent_session_connected_is_called(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)

        await agent._event_bus.dispatch(AgentSessionConnectedEvent())

        listener.on_agent_session_connected.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_agent_interrupted_is_called(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)

        await agent._event_bus.dispatch(AssistantInterruptedEvent())

        listener.on_agent_interrupted.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_agent_error_is_called_with_error(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)
        error = AgentError(type="internal_error", message="oops")

        await agent._event_bus.dispatch(AgentErrorEvent(error=error))

        listener.on_agent_error.assert_called_once_with(error)

    @pytest.mark.asyncio
    async def test_on_user_started_speaking_is_called(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)

        await agent._event_bus.dispatch(UserStartedSpeakingEvent())

        listener.on_user_started_speaking.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_user_stopped_speaking_is_called(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)

        await agent._event_bus.dispatch(UserStoppedSpeakingEvent())

        listener.on_user_stopped_speaking.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_assistant_started_responding_is_called(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)

        await agent._event_bus.dispatch(AssistantStartedRespondingEvent())

        listener.on_assistant_started_responding.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_assistant_stopped_responding_is_called(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)

        await agent._event_bus.dispatch(AssistantStoppedRespondingEvent())

        listener.on_assistant_stopped_responding.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_listener_events_do_not_raise(self) -> None:
        agent = make_agent()
        await agent._event_bus.dispatch(UserStartedSpeakingEvent())
        await agent._event_bus.dispatch(UserStoppedSpeakingEvent())


class TestInactivityTimeoutHandler:
    @pytest.mark.asyncio
    async def test_inactivity_timeout_triggers_stop(self) -> None:
        agent = make_agent()

        await agent._on_inactivity_timeout(
            UserInactivityTimeoutEvent(timeout_seconds=30.0)
        )
        await asyncio.sleep(0)

        assert agent._stop_called is True


class TestListenerCountdownWarnings:
    def test_overrides_countdown_without_timeout_enabled_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class ListenerWithCountdown(AgentListener):
            async def on_user_inactivity_countdown(self, _: int) -> None:
                pass

        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(listener=ListenerWithCountdown())

        assert any("callback will never fire" in r.message for r in caplog.records)

    def test_timeout_enabled_without_countdown_override_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class ListenerWithoutCountdown(AgentListener):
            pass

        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(
                listener=ListenerWithoutCountdown(),
                inactivity_timeout_seconds=10.0,
            )

        assert any("will be silently ignored" in r.message for r in caplog.records)

    def test_no_warning_when_both_configured_correctly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class ListenerWithCountdown(AgentListener):
            async def on_user_inactivity_countdown(
                self, remaining_seconds: int
            ) -> None:
                pass

        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(
                listener=ListenerWithCountdown(),
                inactivity_timeout_seconds=10.0,
            )

        assert not any("countdown" in r.message for r in caplog.records)


class TestListenerTextModalityWarnings:
    def test_override_delta_without_text_output_modality_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class ListenerWithDelta(AgentListener):
            async def on_assistant_transcript_delta(self, _: str) -> None:
                pass

        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(listener=ListenerWithDelta(), output_modalities=["audio"])

        assert any("on_assistant_transcript_delta" in r.message for r in caplog.records)

    def test_override_delta_with_text_output_modality_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class ListenerWithDelta(AgentListener):
            async def on_assistant_transcript_delta(self, _: str) -> None:
                pass

        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(
                listener=ListenerWithDelta(), output_modalities=["audio", "text"]
            )

        assert not any(
            "on_assistant_transcript_delta" in r.message for r in caplog.records
        )

    def test_no_override_delta_without_text_output_modality_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class ListenerWithoutDelta(AgentListener):
            pass

        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(listener=ListenerWithoutDelta(), output_modalities=["audio"])

        assert not any(
            "on_assistant_transcript_delta" in r.message for r in caplog.records
        )
