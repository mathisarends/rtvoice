import logging

from rtvoice.agent.views import AgentError
from rtvoice.events import EventBus
from rtvoice.events.views import AgentErrorEvent
from rtvoice.realtime.schemas import ErrorEvent

logger = logging.getLogger(__name__)


class ErrorWatchdog:
    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe(ErrorEvent, self._on_error)

    async def _on_error(self, event: ErrorEvent) -> None:
        agent_error = AgentError(
            type=event.error.type,
            message=event.error.message,
        )
        logger.error(
            "OpenAI error: %s (event_id=%s)",
            agent_error,
            event.error.event_id,
        )
        await self._event_bus.dispatch(
            AgentErrorEvent(
                error=agent_error,
                event_id=event.error.event_id,
            )
        )
