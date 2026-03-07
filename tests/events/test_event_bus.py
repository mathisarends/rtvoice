import logging
from dataclasses import dataclass

import pytest

from rtvoice.events.bus import EventBus


@dataclass
class UserCreated:
    name: str


@dataclass
class OrderPlaced:
    order_id: int


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


class TestSubscribe:
    @pytest.mark.asyncio
    async def test_subscribed_handler_is_called_on_dispatch(
        self, bus: EventBus
    ) -> None:
        received = []

        async def handler(event: UserCreated) -> None:
            received.append(event)

        bus.subscribe(UserCreated, handler)
        await bus.dispatch(UserCreated(name="Mathis"))

        assert len(received) == 1

    def test_multiple_handlers_can_subscribe_to_same_event(self, bus: EventBus) -> None:
        async def handler_a(event: UserCreated) -> None: ...
        async def handler_b(event: UserCreated) -> None: ...

        bus.subscribe(UserCreated, handler_a)
        bus.subscribe(UserCreated, handler_b)

        assert len(bus._handlers[UserCreated]) == 2

    def test_same_handler_can_subscribe_to_different_events(
        self, bus: EventBus
    ) -> None:
        async def handler(event) -> None: ...

        bus.subscribe(UserCreated, handler)
        bus.subscribe(OrderPlaced, handler)

        assert handler in bus._handlers[UserCreated]
        assert handler in bus._handlers[OrderPlaced]


class TestUnsubscribe:
    def test_removes_handler(self, bus: EventBus) -> None:
        async def handler(event: UserCreated) -> None: ...

        bus.subscribe(UserCreated, handler)
        bus.unsubscribe(UserCreated, handler)

        assert handler not in bus._handlers.get(UserCreated, [])

    def test_unsubscribe_unknown_event_type_does_not_raise(self, bus: EventBus) -> None:
        async def handler(event: UserCreated) -> None: ...

        bus.unsubscribe(UserCreated, handler)

    def test_unsubscribe_unknown_handler_does_not_raise(self, bus: EventBus) -> None:
        async def handler_a(event: UserCreated) -> None: ...
        async def handler_b(event: UserCreated) -> None: ...

        bus.subscribe(UserCreated, handler_a)
        bus.unsubscribe(UserCreated, handler_b)

        assert handler_a in bus._handlers[UserCreated]

    @pytest.mark.asyncio
    async def test_unsubscribed_handler_is_not_called(self, bus: EventBus) -> None:
        called = []

        async def handler(event: UserCreated) -> None:
            called.append(event)

        bus.subscribe(UserCreated, handler)
        bus.unsubscribe(UserCreated, handler)
        await bus.dispatch(UserCreated(name="Mathis"))

        assert called == []


class TestDispatch:
    @pytest.mark.asyncio
    async def test_returns_dispatched_event(self, bus: EventBus) -> None:
        event = UserCreated(name="Mathis")

        async def handler(e: UserCreated) -> None: ...

        bus.subscribe(UserCreated, handler)
        result = await bus.dispatch(event)

        assert result is event

    @pytest.mark.asyncio
    async def test_returns_event_when_no_handlers_registered(
        self, bus: EventBus
    ) -> None:
        event = UserCreated(name="Mathis")
        result = await bus.dispatch(event)

        assert result is event

    @pytest.mark.asyncio
    async def test_all_handlers_are_called(self, bus: EventBus) -> None:
        calls = []

        async def handler_a(event: UserCreated) -> None:
            calls.append("a")

        async def handler_b(event: UserCreated) -> None:
            calls.append("b")

        bus.subscribe(UserCreated, handler_a)
        bus.subscribe(UserCreated, handler_b)
        await bus.dispatch(UserCreated(name="Mathis"))

        assert sorted(calls) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_handler_receives_correct_event(self, bus: EventBus) -> None:
        received = []

        async def handler(event: UserCreated) -> None:
            received.append(event)

        bus.subscribe(UserCreated, handler)
        event = UserCreated(name="Mathis")
        await bus.dispatch(event)

        assert received[0] is event

    @pytest.mark.asyncio
    async def test_does_not_call_handlers_of_other_event_types(
        self, bus: EventBus
    ) -> None:
        called = []

        async def handler(event: OrderPlaced) -> None:
            called.append(event)

        bus.subscribe(OrderPlaced, handler)
        await bus.dispatch(UserCreated(name="Mathis"))

        assert called == []

    @pytest.mark.asyncio
    async def test_failing_handler_does_not_raise(self, bus: EventBus) -> None:
        async def broken_handler(event: UserCreated) -> None:
            raise RuntimeError("boom")

        bus.subscribe(UserCreated, broken_handler)

        await bus.dispatch(UserCreated(name="Mathis"))

    @pytest.mark.asyncio
    async def test_failing_handler_does_not_prevent_other_handlers(
        self, bus: EventBus
    ) -> None:
        called = []

        async def broken_handler(event: UserCreated) -> None:
            raise RuntimeError("boom")

        async def healthy_handler(event: UserCreated) -> None:
            called.append(event)

        bus.subscribe(UserCreated, broken_handler)
        bus.subscribe(UserCreated, healthy_handler)
        await bus.dispatch(UserCreated(name="Mathis"))

        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_logs_warning_when_no_handlers(self, bus: EventBus, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            await bus.dispatch(UserCreated(name="Mathis"))

        assert any("No handlers" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_logs_error_when_handler_raises(self, bus: EventBus, caplog) -> None:
        async def broken_handler(event: UserCreated) -> None:
            raise RuntimeError("boom")

        bus.subscribe(UserCreated, broken_handler)

        with caplog.at_level(logging.ERROR):
            await bus.dispatch(UserCreated(name="Mathis"))

        assert any("boom" in record.message for record in caplog.records)
