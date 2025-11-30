from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtvoice.events import EventBus
    from rtvoice.mic import MicrophoneCapture
    from rtvoice.realtime.reatlime_client import RealtimeClient
    from rtvoice.sound import AudioPlayer
    from rtvoice.wake_word import WakeWordListener


@dataclass
class VoiceAssistantContext:
    wake_word_listener: WakeWordListener
    audio_capture: MicrophoneCapture
    audio_player: AudioPlayer
    event_bus: EventBus
    realtime_client: RealtimeClient
