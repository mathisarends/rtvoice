import asyncio
import logging
from pathlib import Path
from typing import Generic, Self, TypeVar

from rtvoice.audio import (
    AudioInputDevice,
    AudioOutputDevice,
    AudioSession,
    MicrophoneInput,
    SpeakerOutput,
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
    SubAgentCalledEvent,
    UserInactivityTimeoutEvent,
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
    UserTranscriptCompletedEvent,
)
from rtvoice.mcp import MCPServer
from rtvoice.realtime.schemas import (
    AudioConfig,
    AudioInputConfig,
    AudioOutputConfig,
    InputAudioNoiseReductionConfig,
    InputAudioTranscriptionConfig,
    NoiseReductionType,
    RealtimeSessionConfig,
    SemanticVADConfig,
    ServerVADConfig,
    ToolChoiceMode,
    TurnDetectionConfig,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.subagents import SubAgent
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
    ToolCallingWatchdog,
    TranscriptionWatchdog,
    UserInactivityTimeoutWatchdog,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RealtimeAgent(Generic[T]):
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
        subagents: list[SubAgent] | None = None,
        mcp_servers: list[MCPServer] | None = None,
        audio_input: AudioInputDevice | None = None,
        audio_output: AudioOutputDevice | None = None,
        listener: AgentListener | None = None,
        context: T | None = None,
        inactivity_timeout_seconds: float = 10.0,
        recording_path: str | Path | None = None,
        api_key: str | None = None,
    ):
        self._instructions = instructions
        self._model = model
        self._voice = voice
        self._speech_speed = self._clip_speech_speed(speech_speed)
        self._transcription_model = transcription_model
        self._noise_reduction = noise_reduction
        self._turn_detection: TurnDetection = (
            turn_detection if turn_detection is not None else SemanticVAD()
        )

        self._tools = tools.clone() if tools else Tools()
        self._mcp_servers = mcp_servers or []

        self._subagents = subagents or []
        for subagent in self._subagents:
            self._tools.register_subagent(subagent)

        self._listener = listener
        self._inactivity_timeout_seconds = inactivity_timeout_seconds
        self._context = context
        self._recording_path = Path(recording_path) if recording_path else None

        self._stopped = asyncio.Event()
        self._stop_called = False
        self._mcp_ready = asyncio.Event()

        self._event_bus = EventBus()
        self._conversation_history = ConversationHistory(self._event_bus)
        self._tools.set_context(
            SpecialToolParameters(
                event_bus=self._event_bus,
                context=context,
                conversation_history=self._conversation_history,
            )
        )
        self._websocket = RealtimeWebSocket(
            model=self._model, event_bus=self._event_bus, api_key=api_key
        )

        audio_session = AudioSession(
            input_device=audio_input or MicrophoneInput(),
            output_device=audio_output or SpeakerOutput(),
        )

        self._setup_shutdown_handlers()
        self._setup_watchdogs(audio_session)
        self._setup_listener()

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

    def _setup_watchdogs(
        self,
        audio_session: AudioSession,
    ) -> None:
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
        self._user_inactivity_timeout_watchdog = UserInactivityTimeoutWatchdog(
            event_bus=self._event_bus,
            timeout_seconds=self._inactivity_timeout_seconds,
        )
        self._transcription_watchdog = TranscriptionWatchdog(event_bus=self._event_bus)
        self._tool_calling_watchdog = ToolCallingWatchdog(
            event_bus=self._event_bus,
            tools=self._tools,
            websocket=self._websocket,
        )

        self._error_watchdog = ErrorWatchdog(event_bus=self._event_bus)
        self._speech_state_watchdog = SpeechStateWatchdog(event_bus=self._event_bus)

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
            SubAgentCalledEvent,
            lambda e: self._listener.on_subagent_called(e.agent_name, e.task),
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

    async def prepare(self) -> Self:
        own_servers = [self._connect_mcp_servers()]
        subagent_servers = [subagent.prepare() for subagent in self._subagents]

        await asyncio.gather(*own_servers, *subagent_servers, return_exceptions=True)
        return self

    async def run(self) -> AgentResult:
        logger.info("Starting agent...")
        await self.prepare()

        started_at = asyncio.get_event_loop().time()
        session_config = self._build_session_config()
        await self._event_bus.dispatch(StartAgentCommand(session_config=session_config))
        logger.info("Agent started successfully")

        try:
            await self._stopped.wait()
        finally:
            await self.stop()

        return AgentResult(
            turns=self._conversation_history.turns,
            duration_seconds=asyncio.get_event_loop().time() - started_at,
            recording_path=self._recording_path,
        )

    async def _connect_mcp_servers(self) -> None:
        if self._mcp_ready.is_set():
            return

        if not self._mcp_servers:
            self._mcp_ready.set()
            return

        results = await asyncio.gather(
            *[self._connect_mcp_server(s) for s in self._mcp_servers],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error("MCP server connection failed: %s", result)
                continue
            for tool, server in result:
                self._tools.register_mcp(tool, server)

        self._mcp_ready.set()

    async def _connect_mcp_server(self, server: MCPServer) -> list[tuple]:
        await server.connect()
        tools = await server.list_tools()
        logger.info("MCP server connected: %d tools loaded", len(tools))
        return [(tool, server) for tool in tools]

    def _build_session_config(self) -> RealtimeSessionConfig:
        input_config = AudioInputConfig(
            transcription=InputAudioTranscriptionConfig(
                model=self._transcription_model
            ),
            noise_reduction=InputAudioNoiseReductionConfig(
                type=NoiseReductionType(self._noise_reduction)
            ),
            turn_detection=self._build_turn_detection_config(),
        )

        audio_config = AudioConfig(
            output=AudioOutputConfig(
                speed=self._speech_speed,
                voice=self._voice.value,
            ),
            input=input_config,
        )

        return RealtimeSessionConfig(
            model=self._model,
            instructions=self._instructions,
            audio=audio_config,
            tool_choice=ToolChoiceMode.AUTO,
            tools=self._tools.get_tool_schema(),
        )

    def _build_turn_detection_config(self) -> TurnDetectionConfig:
        td = self._turn_detection
        if isinstance(td, SemanticVAD):
            return SemanticVADConfig(eagerness=td.eagerness)
        return ServerVADConfig(
            threshold=td.threshold,
            prefix_padding_ms=td.prefix_padding_ms,
            silence_duration_ms=td.silence_duration_ms,
        )

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
