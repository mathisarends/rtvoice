import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

type EventHandler[E] = Callable[[E], Awaitable[None]]


class EventBus:
    def __init__(self):
        self._handlers: dict[type, list] = {}

    def subscribe[E](self, event_type: type[E], handler: EventHandler[E]) -> None:
        self._handlers.setdefault(event_type, []).append(handler)
        logger.debug(f"Subscribed to {event_type.__name__}")

    def unsubscribe[E](self, event_type: type[E], handler: EventHandler[E]) -> None:
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def dispatch[E](self, event: E) -> E:
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        logger.debug(f"Dispatching {event_type.__name__} to {len(handlers)} handler(s)")

        if not handlers:
            logger.debug(f"No handlers registered for {event_type.__name__}")
            return event

        results = await asyncio.gather(
            *[handler(event) for handler in handlers],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error(
                    f"Handler failed for {event_type.__name__}: {result}",
                    exc_info=result,
                )

        return event
