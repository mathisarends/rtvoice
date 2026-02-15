import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel

from rtvoice.shared.logging import LoggingMixin

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

    async def wait_for_event(
        self,
        event_type: type[T],
        timeout: float | None = None,
        predicate: Callable[[T], bool] | None = None,
    ) -> T:
        future: asyncio.Future[T] = asyncio.Future()

        self.logger.debug(f"Waiting for {event_type.__name__} (timeout={timeout}s)")

        async def handler(event: T) -> None:
            if (predicate is None or predicate(event)) and not future.done():
                future.set_result(event)

        self.subscribe(event_type, handler)

        try:
            if timeout:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future

            self.logger.debug(f"Received {event_type.__name__}")
            return result
        except TimeoutError:
            self.logger.warning(
                f"Timeout waiting for {event_type.__name__} after {timeout}s"
            )
            raise
        finally:
            self.unsubscribe(event_type, handler)
