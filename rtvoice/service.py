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

        self._event_bus = self._create_event_bus()
        self._audio_capture = self._create_audio_capture()
        self._audio_player = self._create_audio_player()
        self._speech_detector = self._create_speech_detector()
        self._wake_word_listener = self._create_wake_word_listener()
        self._sound_event_handler = self._create_sound_event_handler()
        self._realtime_client = self._create_realtime_client()
        self._context = self._create_context()

        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        if self._running:
            self.logger.warning("Agent already running")
            return

        try:
            self.logger.info("Starting Voice Assistant")
            self._running = True

            await self._context.run()

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
            "state_machine": self._cleanup_state_machine(),
            "wake_word": self._cleanup_wake_word_service(),
            "sound": self._cleanup_sound_service(),
        }

        results = await asyncio.gather(*cleanup_tasks.values(), return_exceptions=True)

        for service_name, result in zip(cleanup_tasks.keys(), results, strict=False):
            if isinstance(result, Exception):
                self.logger.exception(
                    "Error cleaning up %s", service_name, exc_info=result
                )

        self.logger.info("All services cleaned up")

    async def _cleanup_state_machine(self) -> None:
        await self._context.state.on_exit(self._context)

    async def _cleanup_wake_word_service(self) -> None:
        if self._wake_word_listener:
            self._wake_word_listener.cleanup()

    async def _cleanup_sound_service(self) -> None:
        self._audio_player.stop_sounds()

    def _create_event_bus(self) -> EventBus:
        event_bus = EventBus()
        event_bus.attach_loop(asyncio.get_running_loop())
        return event_bus

    def _create_audio_capture(self) -> MicrophoneCapture:
        return MicrophoneCapture()

    def _create_audio_player(self) -> AudioPlayer:
        return AudioPlayer(self._voice_settings.playback_strategy)

    def _create_speech_detector(self) -> SpeechDetector:
        return SpeechDetector(
            audio_capture=self._audio_capture,
            event_bus=self._event_bus,
        )

    def _create_wake_word_listener(self) -> WakeWordListener | None:
        return WakeWordListener(
            wakeword=self._wake_word_settings.keyword,
            sensitivity=self._wake_word_settings.sensitivity,
            event_bus=self._event_bus,
        )

    def _create_sound_event_handler(self) -> SoundEventHandler:
        return SoundEventHandler(self._audio_player, self._event_bus)

    def _create_realtime_client(self) -> RealtimeClient:
        self._voice_settings.playback_strategy.set_event_bus(self._event_bus)

        special_tool_parameters = SpecialToolParameters(
            audio_player=self._audio_player,
            event_bus=self._event_bus,
            voice_settings=self._voice_settings,
            tool_calling_model_name=self._model_settings.tool_calling_model_name,
        )

        return RealtimeClient(
            model_settings=self._model_settings,
            voice_settings=self._voice_settings,
            audio_capture=self._audio_capture,
            special_tool_parameters=special_tool_parameters,
            event_bus=self._event_bus,
            tools=self._tools,
        )

    def _create_context(self) -> VoiceAssistantContext:
        return VoiceAssistantContext(
            wake_word_listener=self._wake_word_listener,
            audio_capture=self._audio_capture,
            speech_detector=self._speech_detector,
            audio_player=self._audio_player,
            event_bus=self._event_bus,
            realtime_client=self._realtime_client,
        )
