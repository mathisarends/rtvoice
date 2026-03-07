import asyncio
import logging
from pathlib import Path
from typing import Self

from rtvoice.audio import (
    AudioInputDevice,
    AudioOutputDevice,
    AudioSession,
)
from rtvoice.conversation import ConversationHistory
from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentErrorEvent,
    AgentSessionConnectedEvent,
    AgentStoppedEvent,
    AssistantInterruptedEvent,
    AssistantStartedRespondingEvent,
    AssistantStoppedRespondingEvent,
    AssistantTranscriptCompletedEvent,
    StartAgentCommand,
    UserInactivityTimeoutEvent,
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
    UserTranscriptCompletedEvent,
)
from rtvoice.mcp import MCPServer
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.shared.decorators import timed
from rtvoice.supervisor import SupervisorAgent
from rtvoice.tools import SpecialToolParameters, Tools
from rtvoice.views import (
    AgentListener,
    AgentResult,
    AssistantVoice,
    NoiseReduction,
    RealtimeModel,
    SemanticVAD,
    TranscriptionModel,
    TurnDetection,
)
from rtvoice.watchdogs import (
    AudioPlayerWatchdog,
    AudioRecordingWatchdog,
    ErrorWatchdog,
    InterruptionWatchdog,
    LifecycleWatchdog,
    SpeechStateWatchdog,
    SupervisorInteractionWatchdog,
    ToolCallingWatchdog,
    TranscriptionWatchdog,
    UserInactivityTimeoutWatchdog,
)

logger = logging.getLogger(__name__)


class RealtimeAgent[T]:
    def __init__(
        self,
        instructions: str = "",
        model: RealtimeModel = RealtimeModel.GPT_REALTIME_MINI,
        voice: AssistantVoice = AssistantVoice.MARIN,
        speech_speed: float = 1.0,
        transcription_model: TranscriptionModel = TranscriptionModel.WHISPER_1,
        noise_reduction: NoiseReduction = NoiseReduction.FAR_FIELD,
        turn_detection: TurnDetection | None = None,
        tools: Tools | None = None,
        supervisor_agent: SupervisorAgent | None = None,
        mcp_servers: list[MCPServer] | None = None,
        audio_input: AudioInputDevice | None = None,
        audio_output: AudioOutputDevice | None = None,
        context: T | None = None,
        listener: AgentListener | None = None,
        inactivity_timeout_seconds: float | None = None,
        inactivity_timeout_enabled: bool = False,
        recording_path: str | Path | None = None,
        api_key: str | None = None,
    ):
        if supervisor_agent and mcp_servers:
            logger.warning(
                "mcp_servers are set on RealtimeAgent alongside a supervisor. "
                "Consider attaching MCP servers to the SupervisorAgent instead."
            )

        self._instructions = instructions
        self._model = model
        self._voice = voice
        self._speech_speed = self._clip_speech_speed(speech_speed)
        self._transcription_model = transcription_model
        self._noise_reduction = noise_reduction
        self._turn_detection: TurnDetection = turn_detection or SemanticVAD()
        self._mcp_servers = mcp_servers or []
        self._supervisor_agent = supervisor_agent

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

        self._event_bus = EventBus()
        self._conversation_history = ConversationHistory(self._event_bus)

        self._tools = tools.clone() if tools else Tools()
        self._tools.set_context(
            SpecialToolParameters(
                event_bus=self._event_bus,
                context=context,
                conversation_history=self._conversation_history,
            )
        )
        if self._supervisor_agent:
            self._tools.register_supervisor_agent(self._supervisor_agent)

        self._websocket = RealtimeWebSocket(
            model=self._model, event_bus=self._event_bus, api_key=api_key
        )

        audio_session = AudioSession(
            input_device=audio_input or self._default_audio_input(),
            output_device=audio_output or self._default_audio_output(),
        )

        self._setup_shutdown_handlers()
        self._setup_watchdogs(audio_session)
        self._setup_listener()

    def _default_audio_input(self) -> AudioInputDevice:
        from rtvoice.audio import MicrophoneInput

        return MicrophoneInput()

    def _default_audio_output(self) -> AudioOutputDevice:
        from rtvoice.audio import SpeakerOutput

        return SpeakerOutput()

    def _clip_speech_speed(self, speed: float) -> float:
        clipped = max(0.5, min(speed, 1.5))

        if speed != clipped:
            logger.warning(
                "Speech speed %.2f is out of range [0.5, 1.5], clipping to %.2f",
                speed,
                clipped,
            )

        return clipped

    def _setup_shutdown_handlers(self) -> None:
        self._event_bus.subscribe(
            UserInactivityTimeoutEvent, self._on_inactivity_timeout
        )

    def _setup_watchdogs(self, audio_session: AudioSession) -> None:
        self._audio_player_watchdog = AudioPlayerWatchdog(
            event_bus=self._event_bus,
            session=audio_session,
        )
        self._lifecycle_watchdog = LifecycleWatchdog(
            event_bus=self._event_bus, websocket=self._websocket
        )
        self._interruption_watchdog = InterruptionWatchdog(
            event_bus=self._event_bus,
            websocket=self._websocket,
            session=audio_session,
        )
        self._transcription_watchdog = TranscriptionWatchdog(event_bus=self._event_bus)
        self._tool_calling_watchdog = ToolCallingWatchdog(
            event_bus=self._event_bus,
            tools=self._tools,
            websocket=self._websocket,
        )
        self._error_watchdog = ErrorWatchdog(event_bus=self._event_bus)
        self._speech_state_watchdog = SpeechStateWatchdog(event_bus=self._event_bus)

        if self._should_enable_inactivity_timeout:
            self._user_inactivity_timeout_watchdog = UserInactivityTimeoutWatchdog(
                event_bus=self._event_bus,
                timeout_seconds=self._inactivity_timeout_seconds,
            )

        if self._supervisor_agent:
            self._supervisor_interaction_watchdog = SupervisorInteractionWatchdog(
                event_bus=self._event_bus,
                websocket=self._websocket,
            )

        if self._recording_path:
            self._recording_watchdog = AudioRecordingWatchdog(
                event_bus=self._event_bus,
                output_path=self._recording_path,
            )

    def _setup_listener(self) -> None:
        if not self._listener:
            return

        self._event_bus.subscribe(
            UserTranscriptCompletedEvent,
            lambda e: self._listener.on_user_transcript(e.transcript),
        )
        self._event_bus.subscribe(
            AssistantTranscriptCompletedEvent,
            lambda e: self._listener.on_assistant_transcript(e.transcript),
        )
        self._event_bus.subscribe(
            AgentSessionConnectedEvent,
            lambda _: self._listener.on_agent_session_connected(),
        )
        self._event_bus.subscribe(
            AssistantInterruptedEvent, lambda _: self._listener.on_agent_interrupted()
        )
        self._event_bus.subscribe(
            AgentErrorEvent, lambda e: self._listener.on_agent_error(e.error)
        )
        self._event_bus.subscribe(
            UserStartedSpeakingEvent,
            lambda _: self._listener.on_user_started_speaking(),
        )
        self._event_bus.subscribe(
            UserStoppedSpeakingEvent,
            lambda _: self._listener.on_user_stopped_speaking(),
        )
        self._event_bus.subscribe(
            AssistantStartedRespondingEvent,
            lambda _: self._listener.on_assistant_started_responding(),
        )
        self._event_bus.subscribe(
            AssistantStoppedRespondingEvent,
            lambda _: self._listener.on_assistant_stopped_responding(),
        )

    async def _on_inactivity_timeout(self, event: UserInactivityTimeoutEvent) -> None:
        logger.info(
            "User inactivity timeout after %.1f seconds - triggering shutdown",
            event.timeout_seconds,
        )
        asyncio.ensure_future(self.stop())

    async def run(self) -> AgentResult:
        logger.info("Starting agent...")
        await self.prepare()

        await self._event_bus.dispatch(
            StartAgentCommand(
                model=self._model,
                instructions=self._instructions,
                voice=self._voice,
                speech_speed=self._speech_speed,
                transcription_model=self._transcription_model,
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
    async def prepare(self) -> Self:
        """Prewarms MCP and supervisor connections so the agent starts without delay on run()."""
        tasks = [self._connect_mcp_servers()]
        if self._supervisor_agent:
            tasks.append(self._supervisor_agent.prepare())

        await asyncio.gather(*tasks, return_exceptions=True)
        return self

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
