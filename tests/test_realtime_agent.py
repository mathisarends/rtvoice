import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rtvoice.agent import RealtimeAgent
from rtvoice.events.views import (
    AgentErrorEvent,
    AgentSessionConnectedEvent,
    AgentStoppedEvent,
    AssistantInterruptedEvent,
    AssistantStartedRespondingEvent,
    AssistantStoppedRespondingEvent,
    AssistantTranscriptCompletedEvent,
    AssistantTranscriptDeltaEvent,
    SubAgentFinishedEvent,
    SubAgentStartedEvent,
    UserInactivityTimeoutEvent,
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
    UserTranscriptCompletedEvent,
)
from rtvoice.listener import AgentListener
from rtvoice.views import (
    AgentError,
    AssistantVoice,
    NoiseReduction,
    RealtimeModel,
    SemanticVAD,
    ServerVAD,
    TranscriptionModel,
)

AgentErrorEvent.model_rebuild(_types_namespace={"AgentError": AgentError})


def make_agent(**kwargs) -> RealtimeAgent:
    audio_input = MagicMock()
    audio_output = MagicMock()
    with (
        patch("rtvoice.agent.RealtimeWebSocket"),
        patch("rtvoice.agent.OpenAIProvider"),
    ):
        return RealtimeAgent(
            audio_input=audio_input,
            audio_output=audio_output,
            **kwargs,
        )


class TestInitDefaults:
    def test_default_model_is_mini(self) -> None:
        agent = make_agent()
        assert agent._model == RealtimeModel.GPT_REALTIME_MINI

    def test_default_voice_is_marin(self) -> None:
        agent = make_agent()
        assert agent._voice == AssistantVoice.MARIN

    def test_default_instructions_are_empty(self) -> None:
        agent = make_agent()
        assert agent._instructions == ""

    def test_default_noise_reduction_is_far_field(self) -> None:
        agent = make_agent()
        assert agent._noise_reduction == NoiseReduction.FAR_FIELD

    def test_default_turn_detection_is_semantic_vad(self) -> None:
        agent = make_agent()
        assert isinstance(agent._turn_detection, SemanticVAD)

    def test_default_transcription_model_is_whisper(self) -> None:
        agent = make_agent()
        assert agent._transcription_model == TranscriptionModel.WHISPER_1

    def test_default_inactivity_timeout_disabled(self) -> None:
        agent = make_agent()
        assert agent._should_enable_inactivity_timeout is False

    def test_custom_turn_detection_is_stored(self) -> None:
        vad = ServerVAD(silence_duration_ms=800)
        agent = make_agent(turn_detection=vad)
        assert agent._turn_detection == vad

    def test_custom_model_is_stored(self) -> None:
        agent = make_agent(model=RealtimeModel.GPT_REALTIME)
        assert agent._model == RealtimeModel.GPT_REALTIME

    def test_recording_path_is_converted_to_path_object(self, tmp_path) -> None:
        agent = make_agent(recording_path=str(tmp_path / "rec.wav"))
        from pathlib import Path

        assert agent._recording_path == Path(tmp_path / "rec.wav")

    def test_recording_path_none_when_not_provided(self) -> None:
        agent = make_agent()
        assert agent._recording_path is None

    def test_stop_not_called_initially(self) -> None:
        agent = make_agent()
        assert agent._stop_called is False

    def test_text_output_mode_enables_transcription_watchdog_without_stt(self) -> None:
        agent = make_agent(transcription_model=None, output_modalities=["text"])
        assert hasattr(agent, "_transcription_watchdog")


class TestSpeechSpeedClipping:
    def test_value_within_range_is_unchanged(self) -> None:
        agent = make_agent(speech_speed=1.0)
        assert agent._speech_speed == 1.0

    def test_speech_speed_below_minimum_is_clipped_to_minimum(self) -> None:
        agent = make_agent(speech_speed=0.1)
        assert agent._speech_speed == 0.25

    def test_value_above_maximum_is_clipped_to_one_point_five(self) -> None:
        agent = make_agent(speech_speed=2.0)
        assert agent._speech_speed == 1.5

    def test_exact_minimum_is_not_clipped(self) -> None:
        agent = make_agent(speech_speed=0.5)
        assert agent._speech_speed == 0.5

    def test_exact_maximum_is_not_clipped(self) -> None:
        agent = make_agent(speech_speed=1.5)
        assert agent._speech_speed == 1.5

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
    def test_inactivity_seconds_without_enabled_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(
                inactivity_timeout_seconds=30.0, inactivity_timeout_enabled=False
            )
        assert any(
            "inactivity_timeout_enabled is False" in r.message for r in caplog.records
        )

    def test_no_warning_when_inactivity_fully_disabled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(inactivity_timeout_enabled=False)
        assert not any(
            "inactivity_timeout_enabled is False" in r.message for r in caplog.records
        )

    def test_transcription_none_with_supervisor_defaults_to_whisper(self) -> None:
        supervisor = MagicMock()
        supervisor.name = "supervisor"
        supervisor.description = "A supervisor"
        supervisor.handoff_instructions = None
        supervisor.result_instructions = None
        supervisor.holding_instruction = None
        agent = make_agent(transcription_model=None, subagents=[supervisor])
        assert agent._transcription_model == TranscriptionModel.WHISPER_1

    def test_transcription_none_with_supervisor_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        supervisor = MagicMock()
        supervisor.name = "supervisor"
        supervisor.description = "A supervisor"
        supervisor.handoff_instructions = None
        supervisor.result_instructions = None
        supervisor.holding_instruction = None
        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(transcription_model=None, subagents=[supervisor])
        assert any("Transcription is required" in r.message for r in caplog.records)

    def test_mcp_servers_with_subagents_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        supervisor = MagicMock()
        supervisor.name = "supervisor"
        supervisor.description = "A supervisor"
        supervisor.handoff_instructions = None
        supervisor.result_instructions = None
        supervisor.holding_instruction = None
        mcp_server = MagicMock()
        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(subagents=[supervisor], mcp_servers=[mcp_server])
        assert any("mcp_servers are set" in r.message for r in caplog.records)


class TestInactivityTimeoutFlag:
    def test_enabled_when_both_flag_and_seconds_are_set(self) -> None:
        agent = make_agent(
            inactivity_timeout_seconds=30.0, inactivity_timeout_enabled=True
        )
        assert agent._should_enable_inactivity_timeout is True

    def test_disabled_when_only_flag_is_set_without_seconds(self) -> None:
        agent = make_agent(inactivity_timeout_enabled=True)
        assert agent._should_enable_inactivity_timeout is False

    def test_disabled_when_only_seconds_is_set_without_flag(self) -> None:
        agent = make_agent(inactivity_timeout_seconds=30.0)
        assert agent._should_enable_inactivity_timeout is False


class TestStop:
    @pytest.mark.asyncio
    async def test_dispatches_agent_stopped_event(self) -> None:
        agent = make_agent()
        received = []

        async def capture(e: AgentStoppedEvent) -> None:
            received.append(e)

        agent._event_bus.subscribe(AgentStoppedEvent, capture)

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

        agent._event_bus.subscribe(AgentStoppedEvent, capture)

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

    @pytest.mark.asyncio
    async def test_cleans_up_mcp_servers(self) -> None:
        mcp_server = AsyncMock()
        agent = make_agent(mcp_servers=[mcp_server])

        await agent.stop()

        mcp_server.cleanup.assert_called_once()


class TestPrepare:
    @pytest.mark.asyncio
    async def test_returns_none(self) -> None:
        agent = make_agent()
        result = await agent.prewarm()
        assert result is None

    @pytest.mark.asyncio
    async def test_prepare_is_idempotent_for_mcp_servers(self) -> None:
        mcp_server = AsyncMock()
        mcp_server.list_tools.return_value = []
        agent = make_agent(mcp_servers=[mcp_server])

        await agent.prewarm()
        await agent.prewarm()

        mcp_server.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connects_mcp_server_on_prepare(self) -> None:
        mcp_server = AsyncMock()
        mcp_server.list_tools.return_value = []
        agent = make_agent(mcp_servers=[mcp_server])

        await agent.prewarm()

        mcp_server.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepares_subagents(self) -> None:
        supervisor = MagicMock()
        supervisor.name = "helper"
        supervisor.description = "Helps"
        supervisor.handoff_instructions = None
        supervisor.result_instructions = None
        supervisor.holding_instruction = None
        supervisor.prewarm = AsyncMock()
        other = MagicMock()
        other.name = "other"
        other.description = "Other"
        other.handoff_instructions = None
        other.result_instructions = None
        other.holding_instruction = None
        other.prewarm = AsyncMock()
        agent = make_agent(subagents=[supervisor, other])

        await agent.prewarm()

        supervisor.prewarm.assert_called_once()
        other.prewarm.assert_called_once()


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
    async def test_on_subagent_started_is_called_with_agent_name(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)

        await agent._event_bus.dispatch(SubAgentStartedEvent(agent_name="research"))

        listener.on_subagent_started.assert_called_once_with("research")

    @pytest.mark.asyncio
    async def test_on_subagent_finished_is_called_with_agent_name(self) -> None:
        listener = AsyncMock(spec=AgentListener)
        agent = make_agent(listener=listener)

        await agent._event_bus.dispatch(SubAgentFinishedEvent(agent_name="research"))

        listener.on_subagent_finished.assert_called_once_with("research")

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
            make_agent(
                listener=ListenerWithCountdown(), inactivity_timeout_enabled=False
            )

        assert any("callback will never fire" in r.message for r in caplog.records)

    def test_timeout_enabled_without_countdown_override_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class ListenerWithoutCountdown(AgentListener):
            pass

        with caplog.at_level(logging.WARNING, logger="rtvoice.service"):
            make_agent(
                listener=ListenerWithoutCountdown(),
                inactivity_timeout_enabled=True,
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
                inactivity_timeout_enabled=True,
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
