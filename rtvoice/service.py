import asyncio

from rtvoice.config import AgentEnv
from rtvoice.config.models import (
    ModelSettings,
    TranscriptionSettings,
    VoiceSettings,
    WakeWordSettings,
)
from rtvoice.events import EventBus
from rtvoice.events.schemas import (
    AssistantVoice,
    NoiseReductionType,
    RealtimeModel,
    TranscriptionModel,
)
from rtvoice.mic import MicrophoneCapture, SpeechDetector
from rtvoice.realtime.reatlime_client import RealtimeClient
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.sound import AudioPlayer, SoundEventHandler
from rtvoice.sound.audio import AudioStrategy, PyAudioStrategy
from rtvoice.state.context import VoiceAssistantContext
from rtvoice.state.state_machine import VoiceAssistantStateMachine
from rtvoice.tools import SpecialToolParameters, Tools
from rtvoice.tools.models import SpecialToolParameters as _SpecialToolParameters
from rtvoice.wake_word import WakeWordListener
from rtvoice.wake_word.models import PorcupineWakeWord

_SpecialToolParameters.model_rebuild()


class Agent(LoggingMixin):
    def __init__(
        self,
        instructions: str = "",
        model: RealtimeModel = RealtimeModel.GPT_REALTIME_MINI,
        voice: AssistantVoice = AssistantVoice.MARIN,
        speech_speed: float = 1.0,
        temperature: float = 0.8,
        playback_strategy: AudioStrategy | None = None,
        enable_transcription: bool = False,
        transcription_model: TranscriptionModel = TranscriptionModel.WHISPER_1,
        transcription_language: str | None = None,
        noise_reduction: NoiseReductionType | None = None,
        wake_word: PorcupineWakeWord = PorcupineWakeWord.PICOVOICE,
        wake_word_sensitivity: float = 0.7,
        tools: Tools | None = None,
        tool_calling_model: str | None = None,
        model_settings: ModelSettings | None = None,
        voice_settings: VoiceSettings | None = None,
        transcription_settings: TranscriptionSettings | None = None,
        wake_word_settings: WakeWordSettings | None = None,
        env: AgentEnv | None = None,
    ):
        self._env = env or AgentEnv()

        self._model_settings = model_settings or ModelSettings(
            model=model,
            instructions=instructions,
            temperature=temperature,
            tool_calling_model_name=tool_calling_model,
        )

        self._voice_settings = voice_settings or VoiceSettings(
            voice=voice,
            speech_speed=speech_speed,
            playback_strategy=playback_strategy or PyAudioStrategy(),
        )

        self._transcription_settings = transcription_settings or TranscriptionSettings(
            enabled=enable_transcription,
            model=transcription_model,
            language=transcription_language,
            noise_reduction_mode=noise_reduction,
        )

        self._wake_word_settings = wake_word_settings or WakeWordSettings(
            keyword=wake_word,
            sensitivity=wake_word_sensitivity,
        )

        self._tools = tools or Tools()
        self._tools.mcp_tools = self._model_settings.mcp_tools

        self._state_machine = self._create_state_machine()

        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        if self._running:
            self.logger.warning("Agent already running")
            return

        try:
            self.logger.info("Starting Voice Assistant")
            self._running = True

            await self._state_machine.run()

            while self._running:
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.1)
                    break
                except TimeoutError:
                    continue
                except KeyboardInterrupt:
                    self.logger.info("Shutdown requested by user")
                    break

        except KeyboardInterrupt:
            pass
        finally:
            await self._cleanup_all_services()

    async def stop(self) -> None:
        if not self._running:
            return

        self.logger.info("Stopping Voice Assistant")
        self._running = False
        self._shutdown_event.set()

    async def _cleanup_all_services(self) -> None:
        self.logger.info("Cleaning up all services...")

        cleanup_tasks = {
            "state_machine": self._state_machine.stop(),
        }

        results = await asyncio.gather(*cleanup_tasks.values(), return_exceptions=True)

        for service_name, result in zip(cleanup_tasks.keys(), results, strict=False):
            if isinstance(result, Exception):
                self.logger.exception(
                    "Error cleaning up %s", service_name, exc_info=result
                )

        self.logger.info("All services cleaned up")

    def _create_state_machine(self) -> VoiceAssistantStateMachine:
        event_bus = self._create_event_bus()
        audio_capture = self._create_audio_capture()
        audio_player = self._create_audio_player()
        speech_detector = self._create_speech_detector(audio_capture, event_bus)
        wake_word_listener = self._create_wake_word_listener(event_bus)
        _ = self._create_sound_event_handler(audio_player, event_bus)
        realtime_client = self._create_realtime_client(
            audio_capture, audio_player, event_bus
        )

        context = VoiceAssistantContext(
            wake_word_listener=wake_word_listener,
            audio_capture=audio_capture,
            speech_detector=speech_detector,
            audio_player=audio_player,
            event_bus=event_bus,
            realtime_client=realtime_client,
        )

        return VoiceAssistantStateMachine(context)

    def _create_event_bus(self) -> EventBus:
        event_bus = EventBus()
        event_bus.attach_loop(asyncio.get_running_loop())
        return event_bus

    def _create_audio_capture(self) -> MicrophoneCapture:
        return MicrophoneCapture()

    def _create_audio_player(self) -> AudioPlayer:
        return AudioPlayer(self._voice_settings.playback_strategy)

    def _create_speech_detector(
        self, audio_capture: MicrophoneCapture, event_bus: EventBus
    ) -> SpeechDetector:
        return SpeechDetector(
            audio_capture=audio_capture,
            event_bus=event_bus,
        )

    def _create_wake_word_listener(
        self, event_bus: EventBus
    ) -> WakeWordListener | None:
        return WakeWordListener(
            wakeword=self._wake_word_settings.keyword,
            sensitivity=self._wake_word_settings.sensitivity,
            event_bus=event_bus,
        )

    def _create_sound_event_handler(
        self, audio_player: AudioPlayer, event_bus: EventBus
    ) -> SoundEventHandler:
        return SoundEventHandler(audio_player, event_bus)

    def _create_realtime_client(
        self,
        audio_capture: MicrophoneCapture,
        audio_player: AudioPlayer,
        event_bus: EventBus,
    ) -> RealtimeClient:
        self._voice_settings.playback_strategy.set_event_bus(event_bus)

        special_tool_parameters = SpecialToolParameters(
            audio_player=audio_player,
            event_bus=event_bus,
            voice_settings=self._voice_settings,
            tool_calling_model_name=self._model_settings.tool_calling_model_name,
        )

        return RealtimeClient(
            model_settings=self._model_settings,
            voice_settings=self._voice_settings,
            audio_capture=audio_capture,
            special_tool_parameters=special_tool_parameters,
            event_bus=event_bus,
            tools=self._tools,
        )
