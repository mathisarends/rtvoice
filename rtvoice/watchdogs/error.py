import logging

from rtvoice.events import EventBus
from rtvoice.events.views import AgentErrorEvent
from rtvoice.realtime.schemas import ErrorEvent

logger = logging.getLogger(__name__)


class ErrorWatchdog:
    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe(ErrorEvent, self._on_error)

    async def _on_error(self, event: ErrorEvent) -> None:
        logger.error(
            "OpenAI error [%s] %s (code=%s, param=%s, event_id=%s)",
            event.error.type,
            event.error.message,
            event.error.code,
            event.error.param,
            event.error.event_id,
        )
        await self._event_bus.dispatch(
            AgentErrorEvent(
                type=event.error.type,
                message=event.error.message,
                code=event.error.code,
                param=event.error.param,
                event_id=event.error.event_id,
            )
        )
