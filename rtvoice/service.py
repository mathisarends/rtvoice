import asyncio
from typing import Self

from rtvoice.audio import (
    AudioInputDevice,
    AudioOutputDevice,
    MicrophoneInput,
    SpeakerOutput,
)
from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentStartedEvent,
    AgentStoppedEvent,
    ConversationHistoryResponseEvent,
)
from rtvoice.realtime.schemas import (
    AudioConfig,
    AudioFormat,
    AudioFormatConfig,
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
)
from rtvoice.watchdogs import (
    AudioInputWatchdog,
    AudioOutputWatchdog,
    ConversationHistoryWatchdog,
    MessageTruncationWatchdog,
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
        recording_output_path: str | None = None,
        api_key: str | None = None,
        audio_input: AudioInputDevice | None = None,
        audio_output: AudioOutputDevice | None = None,
    ):
        self._instructions = instructions
        self._model = model
        self._transcription_model = transcription_model
        self._voice = voice
        self._speech_speed = self._clip_speech_speed(speech_speed)
        self._tools = tools or Tools()
        self._stopped = asyncio.Event()

        self._event_bus = EventBus()
        self._websocket = RealtimeWebSocket(
            model=self._model, event_bus=self._event_bus, api_key=api_key
        )

        audio_input_device = audio_input or MicrophoneInput()
        audio_output_device = audio_output or SpeakerOutput()

        self._audio_input_watchdog = AudioInputWatchdog(
            event_bus=self._event_bus,
            device=audio_input_device,
        )
        self._audio_output_watchdog = AudioOutputWatchdog(
            event_bus=self._event_bus,
            device=audio_output_device,
        )
        self._realtime_watchdog = RealtimeWatchdog(
            event_bus=self._event_bus, websocket=self._websocket
        )
        self._message_truncation_watchdog = MessageTruncationWatchdog(
            event_bus=self._event_bus
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
        )
        self._recording_watchdog = RecordingWatchdog(
            event_bus=self._event_bus, output_path=recording_output_path
        )

    def _clip_speech_speed(self, speed: float) -> float:
        clipped = max(0.5, min(speed, 1.5))

        if speed != clipped:
            self.logger.warning(
                "Speech speed %.2f is out of range [0.5, 1.5], clipping to %.2f",
                speed,
                clipped,
            )

        return clipped

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    async def start(self) -> None:
        self.logger.info("Starting agent...")

        session_config = self._build_session_config()
        event = AgentStartedEvent(session_config=session_config)

        await self._event_bus.dispatch(event)
        self.logger.info("Agent started successfully")

        # Blockiert bis stop() aufgerufen wird
        await self._stopped.wait()

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
                format=AudioFormatConfig(type=AudioFormat.PCM16),
                speed=self._speech_speed,
                voice=self._voice.value,
            ),
            input=input_config,
        )

        return RealtimeSessionConfig(
            model=self._model,
            instructions=self._instructions,
            voice=self._voice,
            audio=audio_config,
            tool_choice=ToolChoiceMode.AUTO,
            tools=self._tools.get_schema(),
        )

    async def stop(self) -> AgentHistory:
        self.logger.info("Stopping agent...")

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
