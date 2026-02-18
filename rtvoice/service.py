import asyncio
from typing import Self

from rtvoice.audio import (
    AudioInputDevice,
    AudioOutputDevice,
    AudioSession,
    MicrophoneInput,
    SpeakerOutput,
)
from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentStartedEvent,
    AgentStoppedEvent,
    AssistantTranscriptChunkReceivedEvent,
    AssistantTranscriptCompletedEvent,
    ConversationHistoryResponseEvent,
    StopAgentCommand,
    UserInactivityTimeoutEvent,
    UserTranscriptChunkReceivedEvent,
    UserTranscriptCompletedEvent,
)
from rtvoice.mcp import MCPServer
from rtvoice.realtime.schemas import (
    AudioConfig,
    AudioInputConfig,
    AudioOutputConfig,
    InputAudioTranscriptionConfig,
    RealtimeSessionConfig,
    ToolChoiceMode,
)
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.shared.logging import LoggingMixin
from rtvoice.tools import Tools
from rtvoice.views import (
    AgentHistory,
    AssistantVoice,
    RealtimeModel,
    TranscriptionModel,
    TranscriptListener,
)
from rtvoice.watchdogs import (
    AudioWatchdog,
    ConversationHistoryWatchdog,
    ErrorWatchdog,
    InterruptionWatchdog,
    RealtimeWatchdog,
    RecordingWatchdog,
    ToolCallingWatchdog,
    TranscriptionWatchdog,
    UserInactivityTimeoutWatchdog,
)


class Agent(LoggingMixin):
    def __init__(
        self,
        instructions: str = "",
        model: RealtimeModel = RealtimeModel.GPT_REALTIME_MINI,
        voice: AssistantVoice = AssistantVoice.MARIN,
        speech_speed: float = 1.0,
        transcription_model: TranscriptionModel | None = None,
        tools: Tools | None = None,
        mcp_servers: list[MCPServer] | None = None,
        recording_output_path: str | None = None,
        api_key: str | None = None,
        audio_input: AudioInputDevice | None = None,
        audio_output: AudioOutputDevice | None = None,
        transcript_listener: TranscriptListener | None = None,
    ):
        self._instructions = instructions
        self._model = model
        self._transcription_model = transcription_model
        self._voice = voice
        self._speech_speed = self._clip_speech_speed(speech_speed)
        self._tools = tools or Tools()
        self._mcp_servers = mcp_servers or []
        self._transcript_listener = transcript_listener
        self._stopped = asyncio.Event()

        self._event_bus = EventBus()
        self._websocket = RealtimeWebSocket(
            model=self._model, event_bus=self._event_bus, api_key=api_key
        )

        audio_session = AudioSession(
            input_device=audio_input or MicrophoneInput(),
            output_device=audio_output or SpeakerOutput(),
        )

        self._setup_shutdown_handlers()
        self._setup_watchdogs(audio_session, recording_output_path)
        self._setup_transcript_listener()

    def _clip_speech_speed(self, speed: float) -> float:
        clipped = max(0.5, min(speed, 1.5))

        if speed != clipped:
            self.logger.warning(
                "Speech speed %.2f is out of range [0.5, 1.5], clipping to %.2f",
                speed,
                clipped,
            )

        return clipped

    def _setup_shutdown_handlers(self) -> None:
        self._event_bus.subscribe(StopAgentCommand, self._on_stop_command)
        self._event_bus.subscribe(
            UserInactivityTimeoutEvent, self._on_inactivity_timeout
        )

    def _setup_watchdogs(
        self,
        audio_session: AudioSession,
        recording_output_path: str | None,
    ) -> None:
        self._audio_watchdog = AudioWatchdog(
            event_bus=self._event_bus,
            session=audio_session,
        )
        self._realtime_watchdog = RealtimeWatchdog(
            event_bus=self._event_bus, websocket=self._websocket
        )
        self._interruption_watchdog = InterruptionWatchdog(
            event_bus=self._event_bus, websocket=self._websocket
        )
        self._user_inactivity_timeout_watchdog = UserInactivityTimeoutWatchdog(
            event_bus=self._event_bus
        )
        self._transcription_watchdog = TranscriptionWatchdog(event_bus=self._event_bus)
        self._conversation_history_watchdog = ConversationHistoryWatchdog(
            event_bus=self._event_bus
        )
        self._tool_calling_watchdog = ToolCallingWatchdog(
            event_bus=self._event_bus,
            tool_registry=self._tools.registry,
            websocket=self._websocket,
        )
        self._recording_watchdog = RecordingWatchdog(
            event_bus=self._event_bus, output_path=recording_output_path
        )
        self._error_watchdog = ErrorWatchdog(event_bus=self._event_bus)

    def _setup_transcript_listener(self) -> None:
        if not self._transcript_listener:
            return

        self._event_bus.subscribe(UserTranscriptChunkReceivedEvent, self._on_user_chunk)
        self._event_bus.subscribe(UserTranscriptCompletedEvent, self._on_user_completed)
        self._event_bus.subscribe(
            AssistantTranscriptChunkReceivedEvent, self._on_assistant_chunk
        )
        self._event_bus.subscribe(
            AssistantTranscriptCompletedEvent, self._on_assistant_completed
        )

    async def _on_stop_command(self, event: StopAgentCommand) -> None:
        self.logger.info("Received stop command - triggering shutdown")
        self._stopped.set()

    async def _on_inactivity_timeout(self, event: UserInactivityTimeoutEvent) -> None:
        self.logger.info(
            "User inactivity timeout after %.1f seconds - triggering shutdown",
            event.timeout_seconds,
        )
        self._stopped.set()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    async def start(self) -> None:
        self.logger.info("Starting agent...")

        await self._connect_mcp_servers()

        session_config = self._build_session_config()
        event = AgentStartedEvent(session_config=session_config)

        await self._event_bus.dispatch(event)
        self.logger.info("Agent started successfully")

        # blocked till stop gets called or inactivity timeout occurs
        await self._stopped.wait()

    async def _connect_mcp_servers(self) -> None:
        for server in self._mcp_servers:
            await server.connect()
            tools = await server.list_tools()
            for tool in tools:
                self._tools.register_mcp(tool, server)
            self.logger.info("MCP server connected: %d tools loaded", len(tools))

    def _build_session_config(self) -> RealtimeSessionConfig:
        input_config = AudioInputConfig(
            transcription=(
                InputAudioTranscriptionConfig(model=self._transcription_model)
                if self._transcription_model
                else None
            )
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

    async def stop(self) -> AgentHistory:
        self.logger.info("Stopping agent...")

        for server in self._mcp_servers:
            await server.cleanup()

        event = AgentStoppedEvent()
        await self._event_bus.dispatch(event)

        history_event = await self._event_bus.wait_for_event(
            ConversationHistoryResponseEvent, timeout=5.0
        )

        agent_history = AgentHistory(
            conversation_turns=history_event.conversation_turns
        )

        self._stopped.set()

        self.logger.info("Agent stopped successfully")
        return agent_history

    async def _on_user_chunk(self, event: UserTranscriptChunkReceivedEvent) -> None:
        await self._transcript_listener.on_user_chunk(event.chunk)

    async def _on_user_completed(self, event: UserTranscriptCompletedEvent) -> None:
        await self._transcript_listener.on_user_completed(event.transcript)

    async def _on_assistant_chunk(
        self, event: AssistantTranscriptChunkReceivedEvent
    ) -> None:
        await self._transcript_listener.on_assistant_chunk(event.chunk)

    async def _on_assistant_completed(
        self, event: AssistantTranscriptCompletedEvent
    ) -> None:
        await self._transcript_listener.on_assistant_completed(event.transcript)
