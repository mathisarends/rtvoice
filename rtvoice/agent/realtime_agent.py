import asyncio
import logging
from pathlib import Path

from transitbus import EventBus

from rtvoice.agent.listener import AgentListener, AgentListenerBridge
from rtvoice.agent.subagent import Subagent
from rtvoice.agent.views import (
    AgentResult,
    AssistantVoice,
    InjectedConversation,
    NoiseReduction,
    OutputModality,
    RealtimeModel,
    ReasoningEffort,
    SemanticVAD,
    TranscriptionModel,
    TurnDetection,
)
from rtvoice.audio import (
    AudioInputDevice,
    AudioOutputDevice,
    AudioSession,
    EchoCancellation,
)
from rtvoice.conversation import ConversationHistory
from rtvoice.events.views import (
    AgentStartingEvent,
    AgentStoppedEvent,
    AudioPlaybackCompletedEvent,
    StopAgentCommand,
    UserInactivityTimeoutEvent,
)
from rtvoice.realtime import OpenAIProvider, RealtimeProvider, RealtimeSession
from rtvoice.shared.decorators import timed
from rtvoice.skills import SkillManager, Skills
from rtvoice.tokens import PricingCatalog
from rtvoice.tools import ToolContext, Tools

logger = logging.getLogger(__name__)


class RealtimeAgent[T]:
    def __init__(
        self,
        *,
        instructions: str = "",
        model: RealtimeModel = RealtimeModel.GPT_REALTIME_2_1_MINI,
        reasoning_effort: ReasoningEffort | None = ReasoningEffort.LOW,
        voice: AssistantVoice = AssistantVoice.MARIN,
        speech_speed: float = 1.0,
        transcription_model: TranscriptionModel | None = TranscriptionModel.WHISPER_1,
        output_modalities: list[OutputModality] | None = None,
        noise_reduction: NoiseReduction = NoiseReduction.FAR_FIELD,
        turn_detection: TurnDetection | None = None,
        tools: Tools | None = None,
        skills: Skills | None = None,
        subagent: Subagent | None = None,
        audio_input: AudioInputDevice | None = None,
        audio_output: AudioOutputDevice | None = None,
        enable_echo_cancellation: bool = False,
        context: T | None = None,
        listener: AgentListener | None = None,
        injected_conversation: InjectedConversation | None = None,
        inactivity_timeout_seconds: float | None = None,
        recording_path: str | Path | None = None,
        provider: RealtimeProvider | None = None,
        api_key: str | None = None,
        pricing_catalog: PricingCatalog | None = None,
    ):
        self._subagent = subagent

        if api_key and provider:
            raise ValueError("Pass either `provider` or `api_key`, not both.")

        clipped_speech_speed = self._clip_speech_speed(speech_speed)

        normalized_output_modalities = self._normalize_output_modalities(
            output_modalities
        )
        assistant_text_enabled = "text" in normalized_output_modalities
        effective_turn_detection: TurnDetection = turn_detection or SemanticVAD()

        self._listener = listener
        self._context = context
        recording_path_obj = Path(recording_path) if recording_path else None

        self._stopped = asyncio.Event()
        self._stop_called = False
        self._stop_requested = False

        self._event_bus = EventBus()
        self._conversation_history = ConversationHistory(self._event_bus)

        self._skill_manager = SkillManager(skills) if skills is not None else None
        self._tools = Tools()
        if tools:
            self._tools.merge(tools)

        instructions = (
            self._skill_manager.append_discovery_prompt(instructions)
            if self._skill_manager is not None
            else instructions
        )

        tool_context = ToolContext(
            self._event_bus,
            self._conversation_history,
            context,
            self._skill_manager,
            self._subagent,
        )
        self._tools.set_context(tool_context)

        input_device = audio_input or self._create_default_input()
        output_device = audio_output or self._create_default_output()

        if enable_echo_cancellation:
            input_device, output_device = EchoCancellation().wrap(
                input_device, output_device
            )

        audio_session = AudioSession(
            input_device=input_device,
            output_device=output_device,
        )

        self._realtime_session = RealtimeSession(
            event_bus=self._event_bus,
            model=model,
            reasoning_effort=reasoning_effort,
            instructions=instructions,
            voice=voice,
            speech_speed=clipped_speech_speed,
            transcription_model=transcription_model,
            output_modalities=normalized_output_modalities,
            noise_reduction=noise_reduction,
            turn_detection=effective_turn_detection,
            tools=self._tools,
            audio_session=audio_session,
            injected_conversation=injected_conversation,
            inactivity_timeout_seconds=inactivity_timeout_seconds,
            recording_path=recording_path_obj,
            provider=provider or OpenAIProvider(api_key=api_key),
            pricing_catalog=pricing_catalog,
        )

        self._setup_shutdown_handlers()
        self._listener_bridge: AgentListenerBridge | None = None
        self._setup_listener(
            inactivity_timeout_enabled=inactivity_timeout_seconds is not None,
            assistant_text_enabled=assistant_text_enabled,
        )

    def _create_default_input(self) -> AudioInputDevice:
        from rtvoice.audio import MicrophoneInput

        return MicrophoneInput()

    def _create_default_output(self) -> AudioOutputDevice:
        from rtvoice.audio import SpeakerOutput

        return SpeakerOutput()

    def _clip_speech_speed(self, speed: float) -> float:
        clipped = max(0.25, min(speed, 1.5))

        if speed != clipped:
            logger.warning(
                "Speech speed %.2f is out of range [0.25, 1.5], clipping to %.2f",
                speed,
                clipped,
            )

        return clipped

    def _normalize_output_modalities(
        self, output_modalities: list[OutputModality] | None
    ) -> list[OutputModality]:
        modalities = output_modalities or ["audio"]
        return list(dict.fromkeys(modalities))

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

    async def set_speech_speed(
        self,
        speed: float,
    ) -> None:
        clipped = self._clip_speech_speed(speed)
        await self._realtime_session.update_speech_speed(clipped)

    async def interrupt(self) -> None:
        await self._realtime_session.interrupt()

    async def send_image(self, image_data_url: str, text: str = "") -> None:
        await self._realtime_session.send_image(image_data_url, text)

    @timed()
    async def stop(self) -> None:
        if self._stop_called:
            return
        self._stop_called = True

        logger.info("Stopping agent...")

        await self._event_bus.dispatch(AgentStoppedEvent())

        self._stopped.set()
        logger.info("Agent stopped successfully")

        if self._listener:
            await self._listener.on_agent_stopped()
