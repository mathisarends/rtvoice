from typing import Self

from rtvoice.events import EventBus
from rtvoice.events.views import AgentStartedEvent, AgentStoppedEvent
from rtvoice.realtime.schemas import (
    AudioConfig,
    AudioFormat,
    AudioFormatConfig,
    AudioInputConfig,
    AudioOutputConfig,
    RealtimeSessionConfig,
    ToolChoiceMode,
)
from rtvoice.shared.logging import LoggingMixin
from rtvoice.tools import Tools
from rtvoice.tools.mcp.server import MCPServer
from rtvoice.views import AssistantVoice, RealtimeModel
from rtvoice.watchdogs import (
    MessageTruncationWatchdog,
    RealtimeWatchdog,
    RecordingWatchdog,
    UserInactivityTimeoutWatchdog,
)
from rtvoice.websocket import RealtimeWebSocket


class Agent(LoggingMixin):
    def __init__(
        self,
        instructions: str = "",
        model: RealtimeModel = RealtimeModel.GPT_REALTIME_MINI,
        voice: AssistantVoice = AssistantVoice.MARIN,
        speech_speed: float = 1.0,
        tools: Tools | None = None,
        mcp_servers: list[MCPServer] | None = None,
        recording_output_path: str | None = None,
    ):
        self._instructions = instructions
        self._model = model
        self._voice = voice
        self._speech_speed = self._clip_speech_speed(speech_speed)
        self._tools = tools or Tools()
        self._mcp_servers = mcp_servers or []

        # Initialize infrastructure
        self._event_bus = EventBus()
        self._websocket = RealtimeWebSocket()

        self._realtime_watchdog = RealtimeWatchdog(self._event_bus, self._websocket)
        self._message_truncation_watchdog = MessageTruncationWatchdog(self._event_bus)
        self._user_inactivity_timeout_watchdog = UserInactivityTimeoutWatchdog(
            self._event_bus
        )

        self._recording_watchdog = RecordingWatchdog(
            self._event_bus, recording_output_path
        )

        # self._audio_watchdog = AudioWatchdog(self._event_bus)
        # self._mcp_watchdog = MCPWatchdog(self._event_bus)
        # self._timeout_watchdog = TimeoutWatchdog(self._event_bus)
        # self._tool_watchdog = ToolWatchdog(self._event_bus)

    def _clip_speech_speed(self, speed: float) -> float:
        clipped = max(0.5, min(speed, 1.5))

        if speed != clipped:
            self.logger.warning(
                f"Speech speed {speed} is out of range [0.5, 1.5], clipping to {clipped}"
            )

        return clipped

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    async def start(self) -> None:
        self.logger.info("Starting agent...")

        session_config = self._build_session_config()
        event = AgentStartedEvent(session_config=session_config)

        await self._event_bus.dispatch(event)
        self.logger.info("Agent started successfully")

    def _build_session_config(self) -> RealtimeSessionConfig:
        audio_config = AudioConfig(
            output=AudioOutputConfig(
                format=AudioFormatConfig(type=AudioFormat.PCM16),
                speed=self._speech_speed,
                voice=self._voice.value,
            ),
            input=AudioInputConfig(),
        )

        return RealtimeSessionConfig(
            model=self._model,
            instructions=self._instructions,
            voice=self._voice,
            audio=audio_config,
            tool_choice=ToolChoiceMode.AUTO,
            tools=self._tools.registry.get_openai_schema(),
        )

    async def stop(self) -> None:
        self.logger.info("Stopping agent...")

        event = AgentStoppedEvent()
        await self._event_bus.dispatch(event)

        self.logger.info("Agent stopped successfully")
