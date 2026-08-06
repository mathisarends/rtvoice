import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import assert_never

from transitbus import EventBus

from rtvoice.agent.listener import AgentListener, AgentListenerBridge
from rtvoice.agent.system_prompt import SystemPrompt
from rtvoice.agent.text_agent import TextAgent
from rtvoice.agent.views import (
    AgentResult,
    AssistantVoice,
    InjectedAssistantMessage,
    InjectedConversation,
    InjectedMessage,
    InjectedUserMessage,
    NoiseReduction,
    OutputModality,
    RealtimeModel,
    ReasoningEffort,
    SemanticVAD,
    TranscriptionModel,
    TurnDetection,
)
from rtvoice.audio import (
    AudioInput,
    AudioOutput,
    AudioSession,
    EchoCancellation,
)
from rtvoice.conversation import (
    AssistantTurn,
    ConversationHistory,
    ConversationTurn,
    UserTurn,
)
from rtvoice.events.views import (
    AgentStartingEvent,
    AgentStoppedEvent,
    AudioPlaybackCompletedEvent,
    StopAgentCommand,
    UpdateSpeechSpeedCommand,
    UserInactivityTimeoutEvent,
)
from rtvoice.realtime import (
    OpenAIProvider,
    RealtimeProvider,
    RealtimeSession,
    RealtimeSessionSettings,
)
from rtvoice.shared.decorators import timed
from rtvoice.shared.speech_speed import SpeechSpeed
from rtvoice.skills import Skills
from rtvoice.tokens import PricingCatalog
from rtvoice.tools import ToolContext, Tools

logger = logging.getLogger(__name__)


class RealtimeAgent:
    def __init__(
        self,
        *,
        system_prompt: str,
        model: RealtimeModel = RealtimeModel.GPT_REALTIME_2_1_MINI,
        reasoning_effort: ReasoningEffort | None = ReasoningEffort.LOW,
        voice: AssistantVoice = AssistantVoice.MARIN,
        speech_speed: float = 1.0,
        transcription_model: TranscriptionModel | None = TranscriptionModel.WHISPER_1,
        output_modalities: list[OutputModality] | None = None,
        noise_reduction: NoiseReduction = NoiseReduction.FAR_FIELD,
        turn_detection: TurnDetection | None = None,
        tools: Tools | None = None,
        tool_dependencies: Sequence[object] = (),
        skills: Skills | None = None,
        text_agent: TextAgent | None = None,
        audio_input: AudioInput | None = None,
        audio_output: AudioOutput | None = None,
        echo_cancellation: EchoCancellation | None = None,
        listener: AgentListener | None = None,
        injected_conversation: InjectedConversation | None = None,
        inactivity_timeout_seconds: float | None = None,
        recording_path: str | Path | None = None,
        provider: RealtimeProvider | None = None,
        api_key: str | None = None,
        pricing_catalog: PricingCatalog | None = None,
    ):
        self._text_agent = text_agent

        if api_key and provider:
            raise ValueError("Pass either `provider` or `api_key`, not both.")

        self._listener = listener
        recording_path_obj = Path(recording_path) if recording_path else None

        self._stopped = asyncio.Event()
        self._stop_called = False
        self._stop_requested = False

        self._event_bus = EventBus()
        self._conversation_history = ConversationHistory(self._event_bus)
        if injected_conversation:
            self._conversation_history.seed(
                _injected_turn(message) for message in injected_conversation.messages
            )

        self._skills = skills
        self._tools = Tools()
        if tools:
            self._tools.merge(tools)

        self._system_prompt = SystemPrompt(
            system_prompt,
            skills=self._skills if self._skills is not None else (),
        )

        tool_context = ToolContext(
            self._event_bus,
            self._conversation_history,
            self._skills,
            self._text_agent,
            *tool_dependencies,
        )
        self._tools.set_context(tool_context)

        input_device = audio_input or self._create_default_input()
        output_device = audio_output or self._create_default_output()

        if echo_cancellation:
            input_device, output_device = echo_cancellation.wrap(
                input_device, output_device
            )

        audio_session = AudioSession(
            input_device=input_device,
            output_device=output_device,
        )

        session_settings = RealtimeSessionSettings(
            model=model,
            reasoning_effort=reasoning_effort,
            # Agent-level "system prompt" becomes Session-level "instructions" (the model's own param name).
            instructions=str(self._system_prompt),
            voice=voice,
            speech_speed=SpeechSpeed(speech_speed),
            transcription_model=transcription_model,
            output_modalities=tuple(output_modalities or ["audio"]),
            noise_reduction=noise_reduction,
            turn_detection=turn_detection or SemanticVAD(),
        )

        self._realtime_session = RealtimeSession(
            event_bus=self._event_bus,
            settings=session_settings,
            tools=self._tools,
            audio_session=audio_session,
            provider=provider or OpenAIProvider(api_key=api_key),
            injected_conversation=injected_conversation,
            inactivity_timeout_seconds=inactivity_timeout_seconds,
            recording_path=recording_path_obj,
            pricing_catalog=pricing_catalog,
        )

        self._setup_shutdown_handlers()
        self._listener_bridge: AgentListenerBridge | None = None
        self._setup_listener(
            inactivity_timeout_enabled=inactivity_timeout_seconds is not None,
            assistant_text_enabled=session_settings.assistant_text_enabled,
        )

    def _create_default_input(self) -> AudioInput:
        from rtvoice.audio import MicrophoneInput

        return MicrophoneInput()

    def _create_default_output(self) -> AudioOutput:
        from rtvoice.audio import SpeakerOutput

        return SpeakerOutput()

    def _setup_shutdown_handlers(self) -> None:
        self._event_bus.on(UserInactivityTimeoutEvent, self._on_inactivity_timeout)
        self._event_bus.on(StopAgentCommand, self._on_stop_requested)
        self._event_bus.on(AudioPlaybackCompletedEvent, self._on_playback_completed)

    def _setup_listener(
        self, *, inactivity_timeout_enabled: bool, assistant_text_enabled: bool
    ) -> None:
        if not self._listener:
            return

        self._listener_bridge = AgentListenerBridge(
            event_bus=self._event_bus,
            listener=self._listener,
            inactivity_timeout_enabled=inactivity_timeout_enabled,
            assistant_text_enabled=assistant_text_enabled,
        )
        self._listener_bridge.setup()

    async def _on_stop_requested(self, _: StopAgentCommand) -> None:
        # the stop tool is called while the farewell is still buffered, so defer
        # the teardown until playback drained
        logger.info("Stop requested - shutting down after playback finished")
        self._stop_requested = True

    async def _on_playback_completed(self, _: AudioPlaybackCompletedEvent) -> None:
        if self._stop_requested:
            await self.stop()

    async def _on_inactivity_timeout(self, event: UserInactivityTimeoutEvent) -> None:
        logger.info(
            "User inactivity timeout after %.1f seconds - triggering shutdown",
            event.timeout_seconds,
        )
        asyncio.ensure_future(self.stop())

    async def start(
        self,
    ) -> AgentResult:
        logger.info("Starting agent...")

        await self._event_bus.dispatch(AgentStartingEvent())

        try:
            await self._realtime_session.start()
            logger.info("Agent started successfully")

            await self._stopped.wait()
        finally:
            await self.stop()

        return AgentResult(
            turns=self._conversation_history.turns,
            recording_path=self._realtime_session.recording_path,
            usage=self._realtime_session.usage_report,
        )

    async def set_speech_speed(self, speed: float) -> None:
        await self._event_bus.dispatch(UpdateSpeechSpeedCommand(speed=speed))

    async def interrupt(self) -> None:
        await self._realtime_session.interrupt()

    async def send_message(self, text: str, *, base64_image: str | None = None) -> None:
        sent = await self._realtime_session.send_message(
            text, base64_image=base64_image
        )
        if sent:
            self._conversation_history.add(UserTurn(transcript=text))

    async def send_assistant_message(self, text: str) -> None:
        sent = await self._realtime_session.send_assistant_message(text)
        if sent:
            self._conversation_history.add(AssistantTurn(transcript=text))

    @timed()
    async def stop(self) -> None:
        if self._stop_called:
            return
        self._stop_called = True

        logger.info("Stopping agent...")

        await self._event_bus.dispatch(AgentStoppedEvent())

        self._stopped.set()
        logger.info("Agent stopped successfully")


def _injected_turn(message: InjectedMessage) -> ConversationTurn:
    match message:
        case InjectedUserMessage(text=text):
            return UserTurn(transcript=text)
        case InjectedAssistantMessage(text=text):
            return AssistantTurn(transcript=text)
        case _:
            assert_never(message)
