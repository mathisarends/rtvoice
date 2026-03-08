import logging

from rtvoice.events import EventBus
from rtvoice.realtime.schemas import InputAudioBufferAppendEvent
from rtvoice.realtime.websocket import RealtimeWebSocket

logger = logging.getLogger(__name__)


class AudioForwardWatchdog:
    def __init__(self, event_bus: EventBus, websocket: RealtimeWebSocket):
        self._websocket = websocket
        event_bus.subscribe(
            InputAudioBufferAppendEvent, self._on_input_audio_buffer_append
        )

    async def _on_input_audio_buffer_append(
        self, event: InputAudioBufferAppendEvent
    ) -> None:
        if not self._websocket.is_connected:
            logger.warning("Cannot send audio — WebSocket not connected")
            return
        await self._websocket.send(event)
