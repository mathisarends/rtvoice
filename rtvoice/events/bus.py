import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel

from rtvoice.shared.logging_mixin import LoggingMixin

T = TypeVar("T", bound=BaseModel)
EventHandler = Callable[[T], Awaitable[None]]


class EventBus(LoggingMixin):
    def __init__(self):
        self._handlers: dict[type[BaseModel], list[EventHandler]] = {}

    def subscribe(self, event_type: type[T], handler: EventHandler[T]) -> None:
        self._handlers.setdefault(event_type, []).append(handler)
        self.logger.debug(f"Subscribed to {event_type.__name__}")

    def unsubscribe(self, event_type: type[T], handler: EventHandler[T]) -> None:
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def dispatch(self, event: T) -> T:
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        self.logger.debug(
            f"Dispatching {event_type.__name__} to {len(handlers)} handler(s)"
        )

        if not handlers:
            self.logger.warning(f"No handlers registered for {event_type.__name__}")
            return event

        tasks = [handler(event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self.logger.error(
                    f"Handler failed for {event_type.__name__}: {result}",
                    exc_info=result,
                )

        return event
