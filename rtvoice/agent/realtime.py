import asyncio
import logging
from pathlib import Path

from rtvoice.agent.listener import AgentListener, AgentListenerBridge
from rtvoice.agent.supervisor import (
    Supervisor,
    SupervisorClarificationNeeded,
    SupervisorResult,
)
from rtvoice.agent.views import (
    AgentResult,
    AssistantVoice,
    ClarificationCheckpoint,
    ConversationSeed,
    NoiseReduction,
    OutputModality,
    RealtimeModel,
    SemanticVAD,
    TranscriptionModel,
    TurnDetection,
)
from rtvoice.audio import (
    AudioInputDevice,
    AudioOutputDevice,
    AudioSession,
)
from rtvoice.conversation import ConversationHistory
from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentStartingEvent,
    AgentStoppedEvent,
    UserInactivityTimeoutEvent,
)
from rtvoice.realtime import OpenAIProvider, RealtimeProvider, RealtimeSession
from rtvoice.shared.decorators import timed
from rtvoice.tools import Inject, ToolContext, Tools

logger = logging.getLogger(__name__)


class RealtimeAgent[T]:
    def __init__(
        self,
        *,
        instructions: str = "",
        model: RealtimeModel = RealtimeModel.GPT_REALTIME_MINI,
        voice: AssistantVoice = AssistantVoice.MARIN,
        speech_speed: float = 1.0,
        transcription_model: TranscriptionModel | None = TranscriptionModel.WHISPER_1,
        output_modalities: list[OutputModality] | None = None,
        noise_reduction: NoiseReduction = NoiseReduction.FAR_FIELD,
        turn_detection: TurnDetection | None = None,
        tools: Tools | None = None,
        supervisor: Supervisor | None = None,
        audio_input: AudioInputDevice | None = None,
        audio_output: AudioOutputDevice | None = None,
        context: T | None = None,
        listener: AgentListener | None = None,
        conversation_seed: ConversationSeed | None = None,
        inactivity_timeout_seconds: float | None = None,
        inactivity_timeout_enabled: bool = False,
        recording_path: str | Path | None = None,
        provider: RealtimeProvider | None = None,
        api_key: str | None = None,
    ):
        self._supervisor = supervisor

        if api_key and provider:
            raise ValueError("Pass either `provider` or `api_key`, not both.")

        clipped_speech_speed = self._clip_speech_speed(speech_speed)

        if transcription_model is None and self._supervisor:
            logger.warning(
                "transcription_model is None but a supervisor is attached. "
                "Transcription is required for supervisor handoffs - "
                "defaulting to TranscriptionModel.WHISPER_1."
            )
            transcription_model = TranscriptionModel.WHISPER_1

        normalized_output_modalities = self._normalize_output_modalities(
            output_modalities
        )
        assistant_text_enabled = "text" in normalized_output_modalities
        effective_turn_detection: TurnDetection = turn_detection or SemanticVAD()

        if inactivity_timeout_seconds is not None and not inactivity_timeout_enabled:
            raise ValueError(
                "inactivity_timeout_seconds is set but inactivity_timeout_enabled is False. "
                "Set inactivity_timeout_enabled=True or remove inactivity_timeout_seconds."
            )

        should_enable_inactivity_timeout = (
            inactivity_timeout_enabled and inactivity_timeout_seconds is not None
        )

        self._listener = listener
        self._context = context
        recording_path_obj = Path(recording_path) if recording_path else None

        self._stopped = asyncio.Event()
        self._stop_called = False

        self._event_bus = EventBus()
        self._conversation_history = ConversationHistory(self._event_bus)

        self._tools = Tools()
        if tools:
            self._tools.merge(tools)

        self._tools.set_context(
            ToolContext(
                event_bus=self._event_bus,
                context=context,
                conversation_history=self._conversation_history,
            )
        )
        if self._supervisor:
            self._register_supervisor(self._supervisor)

        audio_session = AudioSession(
            input_device=audio_input or self._create_default_input(),
            output_device=audio_output or self._create_default_output(),
        )

        self._realtime_session = RealtimeSession(
            event_bus=self._event_bus,
            model=model,
            instructions=instructions,
            voice=voice,
            speech_speed=clipped_speech_speed,
            transcription_model=transcription_model,
            output_modalities=normalized_output_modalities,
            noise_reduction=noise_reduction,
            turn_detection=effective_turn_detection,
            tools=self._tools,
            audio_session=audio_session,
            supervisor=self._supervisor,
            conversation_seed=conversation_seed,
            inactivity_timeout_enabled=should_enable_inactivity_timeout,
            inactivity_timeout_seconds=inactivity_timeout_seconds,
            recording_path=recording_path_obj,
            provider=provider or OpenAIProvider(api_key=api_key),
        )

        self._setup_shutdown_handlers()
        self._listener_bridge: AgentListenerBridge | None = None
        self._setup_listener(
            inactivity_timeout_enabled=should_enable_inactivity_timeout,
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

    def _register_supervisor(self, supervisor: Supervisor) -> None:
        description = supervisor.description
        if supervisor.handoff_instructions:
            description = (
                f"{supervisor.description}\n\n"
                f"Handoff instructions: {supervisor.handoff_instructions}"
            )

        self._register_supervisor_handoff(supervisor, description)

    def _register_supervisor_handoff(
        self, supervisor: Supervisor, description: str
    ) -> None:
        paused_for_clarification: ClarificationCheckpoint | None = None

        @self._tools.action(
            description,
            name=supervisor.name,
            result_instruction=supervisor.result_instructions,
            holding_instruction=supervisor.holding_instruction,
        )
        async def _handoff(
            task: str,
            conversation_history: Inject[ConversationHistory],
            clarification_answer: str | None = None,
        ) -> SupervisorResult:
            nonlocal paused_for_clarification

            is_resuming = (
                paused_for_clarification is not None
                and clarification_answer is not None
            )

            if is_resuming:
                checkpoint = paused_for_clarification
                paused_for_clarification = None
                result = await supervisor.resume(
                    clarification_answer=clarification_answer,
                    resume_history=checkpoint.resume_history,
                    clarify_call_id=checkpoint.clarify_call_id,
                )
            else:
                context = (
                    conversation_history.format() if conversation_history else None
                )
                result = await supervisor.run(task, context=context)

            if isinstance(result, SupervisorClarificationNeeded):
                paused_for_clarification = ClarificationCheckpoint(
                    resume_history=result.resume_history,
                    clarify_call_id=result.clarify_call_id,
                )

            return result

    def _setup_shutdown_handlers(self) -> None:
        self._event_bus.subscribe(
            UserInactivityTimeoutEvent, self._on_inactivity_timeout
        )

    def _setup_listener(
        self, *, inactivity_timeout_enabled: bool, assistant_text_enabled: bool
    ) -> None:
        if not self._listener:
            return

        self._listener_bridge = AgentListenerBridge(
            event_bus=self._event_bus,
            listener=self._listener,
            inactivity_timeout_enabled=inactivity_timeout_enabled,
            has_supervisor=bool(self._supervisor),
            assistant_text_enabled=assistant_text_enabled,
        )
        self._listener_bridge.setup()

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
        )

    async def set_speech_speed(
        self,
        speed: float,
    ) -> None:
        clipped = self._clip_speech_speed(speed)
        await self._realtime_session.update_speech_speed(clipped)

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
