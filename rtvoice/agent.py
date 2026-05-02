import asyncio
import logging
from pathlib import Path
from typing import Annotated

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
    StartAgentCommand,
    UpdateSpeechSpeedCommand,
    UserInactivityTimeoutEvent,
)
from rtvoice.handler import (
    AudioForwarder,
    AudioHandler,
    AudioRecorder,
    SpeechStateTracker,
    SubAgentCoordinator,
    ToolCallHandler,
    TranscriptionAccumulator,
)
from rtvoice.listener import AgentListener, AgentListenerBridge
from rtvoice.mcp import MCPServer
from rtvoice.realtime import OpenAIProvider, RealtimeProvider
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.shared.decorators import timed
from rtvoice.subagent import SubAgent
from rtvoice.subagent.views import AgentClarificationNeeded, SubAgentResult
from rtvoice.tools import Inject, ToolContext, Tools
from rtvoice.views import (
    AgentResult,
    AssistantVoice,
    ClarificationCheckpoint,
    NoiseReduction,
    OutputModality,
    RealtimeModel,
    SemanticVAD,
    TranscriptionModel,
    TurnDetection,
)
from rtvoice.watchdogs import (
    ErrorWatchdog,
    InterruptionWatchdog,
    LifecycleWatchdog,
    SessionWatchdog,
    UserInactivityTimeoutWatchdog,
)

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
        subagents: list[SubAgent] | None = None,
        mcp_servers: list[MCPServer] | None = None,
        audio_input: AudioInputDevice | None = None,
        audio_output: AudioOutputDevice | None = None,
        context: T | None = None,
        event_bus: EventBus | None = None,
        listener: AgentListener | None = None,
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

        self._instructions = instructions
        self._model = model
        self._voice = voice
        self._speech_speed = self._clip_speech_speed(speech_speed)

        if transcription_model is None and self._subagents:
            logger.warning(
                "transcription_model is None but subagents are attached. "
                "Transcription is required for subagent handoffs — "
                "defaulting to TranscriptionModel.WHISPER_1."
            )
            transcription_model = TranscriptionModel.WHISPER_1

        self._transcription_model = transcription_model
        self._transcription_enabled = transcription_model is not None
        self._output_modalities = self._normalize_output_modalities(output_modalities)
        self._assistant_text_enabled = "text" in self._output_modalities
        self._noise_reduction = noise_reduction
        self._turn_detection: TurnDetection = turn_detection or SemanticVAD()
        self._mcp_servers = mcp_servers or []

        if inactivity_timeout_seconds is not None and not inactivity_timeout_enabled:
            logger.warning(
                "inactivity_timeout_seconds is set but inactivity_timeout_enabled is False. "
                "The timeout will not be active."
            )

        self._should_enable_inactivity_timeout = (
            inactivity_timeout_enabled and inactivity_timeout_seconds is not None
        )

        self._listener = listener
        self._inactivity_timeout_seconds = inactivity_timeout_seconds
        self._inactivity_timeout_enabled = inactivity_timeout_enabled
        self._context = context
        self._recording_path = Path(recording_path) if recording_path else None

        self._stopped = asyncio.Event()
        self._stop_called = False
        self._mcp_ready = asyncio.Event()

        self._event_bus = event_bus or EventBus()
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
        for subagent in self._subagents:
            self._register_subagent(subagent)

        self._websocket = RealtimeWebSocket(
            model=self._model,
            provider=provider or OpenAIProvider(api_key=api_key),
        )

        audio_session = AudioSession(
            input_device=audio_input or self._default_audio_input(),
            output_device=audio_output or self._default_audio_output(),
        )

        self._setup_shutdown_handlers()
        self._setup_watchdogs(audio_session)
        self._listener_bridge: AgentListenerBridge | None = None
        self._setup_listener()

    def _default_audio_input(self) -> AudioInputDevice:
        from rtvoice.audio import MicrophoneInput

        return MicrophoneInput()

    def _default_audio_output(self) -> AudioOutputDevice:
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
            task: Annotated[
                str,
                "The task or question to delegate to this agent. Be specific and include enough context for the agent to act without clarification.",
            ],
            conversation_history: Inject[ConversationHistory],
            clarification_answer: Annotated[
                str | None,
                "If this is a follow-up call after a clarification request, provide the user's answer here. Leave empty for the initial call.",
            ] = None,
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

    def _setup_watchdogs(self, audio_session: AudioSession) -> None:
        self._audio_handler = AudioHandler(
            event_bus=self._event_bus,
            audio_session=audio_session,
        )
        self._lifecycle_watchdog = LifecycleWatchdog(
            event_bus=self._event_bus, websocket=self._websocket
        )
        self._session_watchdog = SessionWatchdog(
            event_bus=self._event_bus, websocket=self._websocket
        )
        self._audio_forwarder = AudioForwarder(
            event_bus=self._event_bus, websocket=self._websocket
        )
        self._interruption_watchdog = InterruptionWatchdog(
            event_bus=self._event_bus,
            websocket=self._websocket,
            audio_session=audio_session,
        )
        if self._transcription_enabled or self._assistant_text_enabled:
            self._transcription_accumulator = TranscriptionAccumulator(
                event_bus=self._event_bus
            )

        self._tool_call_handler = ToolCallHandler(
            event_bus=self._event_bus,
            tools=self._tools,
            websocket=self._websocket,
            subagent_tool_names={s.name for s in self._subagents} or None,
        )
        if self._subagents:
            self._subagent_coordinator = SubAgentCoordinator(
                event_bus=self._event_bus,
                tools=self._tools,
                websocket=self._websocket,
            )
            for subagent in self._subagents:
                self._subagent_coordinator.register_subagent(subagent.name, subagent)
        self._error_watchdog = ErrorWatchdog(event_bus=self._event_bus)
        self._speech_state_tracker = SpeechStateTracker(event_bus=self._event_bus)

        if self._should_enable_inactivity_timeout:
            self._user_inactivity_timeout_watchdog = UserInactivityTimeoutWatchdog(
                event_bus=self._event_bus,
                timeout_seconds=self._inactivity_timeout_seconds,
            )

        if self._recording_path:
            self._audio_recorder = AudioRecorder(
                event_bus=self._event_bus,
                output_path=self._recording_path,
            )

    def _setup_listener(self) -> None:
        if not self._listener:
            return

        self._listener_bridge = AgentListenerBridge(
            event_bus=self._event_bus,
            listener=self._listener,
            inactivity_timeout_enabled=self._should_enable_inactivity_timeout,
            has_subagents=bool(self._subagents),
            assistant_text_enabled=self._assistant_text_enabled,
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
        """Start the agent and block until the session ends.

        Dispatches a `StartAgentCommand` to kick off audio I/O and the WebSocket
        connection, then waits until `stop()` is called — either manually, via
        inactivity timeout, or through an error watchdog.
        """
        logger.info("Starting agent...")

        await self._event_bus.dispatch(AgentStartingEvent())
        await self.prewarm()

        await self._event_bus.dispatch(
            StartAgentCommand(
                model=self._model,
                instructions=self._instructions,
                voice=self._voice,
                speech_speed=self._speech_speed,
                transcription_model=self._transcription_model,
                output_modalities=self._output_modalities,
                noise_reduction=self._noise_reduction,
                turn_detection=self._turn_detection,
                tools=self._tools,
            )
        )
        logger.info("Agent started successfully")

        try:
            await self._stopped.wait()
        finally:
            await self.stop()

        return AgentResult(
            turns=self._conversation_history.turns,
            recording_path=self._recording_path,
        )

    @timed()
    async def prewarm(self) -> None:
        """Prewarm MCP and subagent connections before `run()`.

        Calling this explicitly avoids a cold-start delay when the session begins.
        Safe to call multiple times — subsequent calls are no-ops for MCP servers
        that are already connected.
        """
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
        """Update the assistant's speech speed mid-session.

        Clamps the value to ``[0.25, 1.5]`` before applying. The change takes
        effect on the next response — audio that is already playing is unaffected.
        """
        clipped = self._clip_speech_speed(speed)
        await self._event_bus.dispatch(UpdateSpeechSpeedCommand(speed=clipped))

    @timed()
    async def stop(self) -> None:
        """Gracefully shut down the agent.

        Cleans up all MCP server connections, dispatches `AgentStoppedEvent`,
        and signals the `run()` coroutine to return. Idempotent — safe to call
        multiple times.
        """
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
