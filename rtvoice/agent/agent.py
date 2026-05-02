import asyncio
import logging
from pathlib import Path

from rtvoice.agent.listener import AgentListener, AgentListenerBridge
from rtvoice.agent.prompts import SystemPrompt
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
from rtvoice.mcp import MCPServer
from rtvoice.realtime import OpenAIProvider, RealtimeProvider, RealtimeSession
from rtvoice.shared.decorators import timed
from rtvoice.subagent import SubAgent
from rtvoice.subagent.views import AgentClarificationNeeded, SubAgentResult
from rtvoice.token import TokenTracker
from rtvoice.tools import Inject, ToolContext, Tools

logger = logging.getLogger(__name__)


class RealtimeAgent[T]:
    def __init__(
        self,
        *,
        extends_system_prompt: str = "",
        override_system_promt: str = "",
        model: RealtimeModel = RealtimeModel.GPT_REALTIME_MINI,
        voice: AssistantVoice = AssistantVoice.MARIN,
        speech_speed: float = 1.0,
        transcription_model: TranscriptionModel | None = TranscriptionModel.WHISPER_1,
        output_modalities: list[OutputModality] | None = None,
        noise_reduction: NoiseReduction = NoiseReduction.FAR_FIELD,
        turn_detection: TurnDetection | None = None,
        tools: Tools | None = None,
        subagents: list[SubAgent] | None = None,
        mcp_servers: list[MCPServer] | None = None,
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
        self._subagents = list(subagents or [])
        self._validate_subagent_names(self._subagents)

        if self._subagents and mcp_servers:
            logger.warning(
                "mcp_servers are set on RealtimeAgent alongside subagents. "
                "Consider attaching MCP servers to subagents instead."
            )

        if api_key and provider:
            raise ValueError("Pass either `provider` or `api_key`, not both.")

        clipped_speech_speed = self._clip_speech_speed(speech_speed)

        if transcription_model is None and self._subagents:
            logger.warning(
                "transcription_model is None but subagents are attached. "
                "Transcription is required for subagent handoffs — "
                "defaulting to TranscriptionModel.WHISPER_1."
            )
            transcription_model = TranscriptionModel.WHISPER_1

        normalized_output_modalities = self._normalize_output_modalities(
            output_modalities
        )
        assistant_text_enabled = "text" in normalized_output_modalities
        effective_turn_detection: TurnDetection = turn_detection or SemanticVAD()
        self._mcp_servers = mcp_servers or []

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
        self._mcp_ready = asyncio.Event()

        self._event_bus = EventBus()
        self._conversation_history = ConversationHistory(self._event_bus)
        self._token_tracker = TokenTracker()

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
        for subagent in self._subagents:
            subagent.use_token_tracker(self._token_tracker)
            self._register_subagent(subagent)

        audio_session = AudioSession(
            input_device=audio_input or self._create_default_input(),
            output_device=audio_output or self._create_default_output(),
        )

        system_prompt = SystemPrompt(
            extends_system_prompt=extends_system_prompt,
            override_syste_Mpromt=override_system_promt,
        )
        instructions = system_prompt.render()

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
            subagents=self._subagents,
            conversation_seed=conversation_seed,
            inactivity_timeout_enabled=should_enable_inactivity_timeout,
            inactivity_timeout_seconds=inactivity_timeout_seconds,
            recording_path=recording_path_obj,
            provider=provider or OpenAIProvider(api_key=api_key),
            token_tracker=self._token_tracker,
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

    def _validate_subagent_names(self, subagents: list[SubAgent]) -> None:
        seen_names: set[str] = set()
        for subagent in subagents:
            if subagent.name in seen_names:
                raise ValueError(
                    f"Duplicate subagent name '{subagent.name}'. "
                    "Subagent names must be unique."
                )
            seen_names.add(subagent.name)

    def _register_subagent(self, subagent: SubAgent) -> None:
        description = subagent.description
        if subagent.handoff_instructions:
            description = (
                f"{subagent.description}\n\n"
                f"Handoff instructions: {subagent.handoff_instructions}"
            )

        subagent_name = subagent.name
        self._register_subagent_handoff(subagent, subagent_name, description)

    def _register_subagent_handoff(
        self, subagent: SubAgent, subagent_name: str, description: str
    ) -> None:
        paused_for_clarification: ClarificationCheckpoint | None = None

        @self._tools.action(
            description,
            name=subagent_name,
            result_instruction=subagent.result_instructions,
            holding_instruction=subagent.holding_instruction,
        )
        async def _handoff(
            task: str,
            conversation_history: Inject[ConversationHistory],
            clarification_answer: str | None = None,
        ) -> SubAgentResult:
            nonlocal paused_for_clarification

            is_resuming = (
                paused_for_clarification is not None
                and clarification_answer is not None
            )

            if is_resuming:
                checkpoint = paused_for_clarification
                paused_for_clarification = None
                result = await subagent.resume(
                    clarification_answer=clarification_answer,
                    resume_history=checkpoint.resume_history,
                    clarify_call_id=checkpoint.clarify_call_id,
                )
            else:
                context = (
                    conversation_history.format() if conversation_history else None
                )
                result = await subagent.run(task, context=context)

            if isinstance(result, AgentClarificationNeeded):
                # Subagent yielded control back to the realtime agent to collect user input
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
            has_subagents=bool(self._subagents),
            assistant_text_enabled=assistant_text_enabled,
        )
        self._listener_bridge.setup()

    async def _on_inactivity_timeout(self, event: UserInactivityTimeoutEvent) -> None:
        logger.info(
            "User inactivity timeout after %.1f seconds - triggering shutdown",
            event.timeout_seconds,
        )
        asyncio.ensure_future(self.stop())

    async def run(
        self,
    ) -> AgentResult:
        logger.info("Starting agent...")

        await self._event_bus.dispatch(AgentStartingEvent())
        await self.prewarm()

        try:
            await self._realtime_session.start()
            logger.info("Agent started successfully")

            await self._stopped.wait()
        finally:
            await self.stop()

        return AgentResult(
            turns=self._conversation_history.turns,
            recording_path=self._realtime_session.recording_path,
            token_usage=self._token_tracker.summary(),
        )

    @timed()
    async def prewarm(self) -> None:
        tasks = [self._connect_mcp_servers()]
        tasks.extend(subagent.prewarm() for subagent in self._subagents)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _connect_mcp_servers(self) -> None:
        if self._mcp_ready.is_set() or not self._mcp_servers:
            self._mcp_ready.set()
            return

        results = await asyncio.gather(
            *[self._connect_and_register_mcp_server(s) for s in self._mcp_servers],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error("MCP server connection failed: %s", result)

        self._mcp_ready.set()

    async def _connect_and_register_mcp_server(self, server: MCPServer) -> None:
        await server.connect()
        tools = await server.list_tools()
        for tool in tools:
            self._tools.register_mcp(tool, server)
        logger.info("MCP server connected: %d tools loaded", len(tools))

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

        for server in self._mcp_servers:
            await server.cleanup()

        await self._event_bus.dispatch(AgentStoppedEvent())

        self._stopped.set()
        logger.info("Agent stopped successfully")

        if self._listener:
            await self._listener.on_agent_stopped()
