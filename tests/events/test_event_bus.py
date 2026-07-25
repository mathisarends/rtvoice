import pytest
from transitbus import Dispatch, Event, EventBus


class UserCreated(Event):
    name: str


class OrderPlaced(Event):
    order_id: int


@pytest.fixture
def bus() -> EventBus:
    return EventBus(name="test")


class TestSubscriptions:
    @pytest.mark.asyncio
    async def test_registered_handler_is_called(self, bus: EventBus) -> None:
        received: list[UserCreated] = []

        async def handler(event: UserCreated) -> None:
            received.append(event)

        bus.on(UserCreated, handler)
        event = UserCreated(name="Mathis")
        await bus.dispatch(event)

        assert received == [event]

    @pytest.mark.asyncio
    async def test_off_removes_handler(self, bus: EventBus) -> None:
        received: list[UserCreated] = []

        async def handler(event: UserCreated) -> None:
            received.append(event)

        bus.on(UserCreated, handler)
        bus.off(UserCreated, handler)
        await bus.dispatch(UserCreated(name="Mathis"))

        assert received == []

    @pytest.mark.asyncio
    async def test_only_matching_event_handlers_are_called(self, bus: EventBus) -> None:
        received: list[OrderPlaced] = []

        async def handler(event: OrderPlaced) -> None:
            received.append(event)

        bus.on(OrderPlaced, handler)
        await bus.dispatch(UserCreated(name="Mathis"))

        assert received == []


class TestDispatch:
    @pytest.mark.asyncio
    async def test_returns_dispatch_handle_for_event(self, bus: EventBus) -> None:
        event = UserCreated(name="Mathis")

        handle = bus.dispatch(event)
        completed = await handle

        assert isinstance(handle, Dispatch)
        assert completed is handle
        assert handle.event is event
        assert handle.done

    @pytest.mark.asyncio
    async def test_handlers_run_in_registration_order(self, bus: EventBus) -> None:
        calls: list[str] = []

        async def handler_a(_: UserCreated) -> None:
            calls.append("a")

        async def handler_b(_: UserCreated) -> None:
            calls.append("b")

        bus.on(UserCreated, handler_a)
        bus.on(UserCreated, handler_b)
        await bus.dispatch(UserCreated(name="Mathis"))

        assert calls == ["a", "b"]

    @pytest.mark.asyncio
    async def test_handler_failure_is_captured_and_does_not_stop_dispatch(
        self, bus: EventBus
    ) -> None:
        received: list[UserCreated] = []

        async def broken_handler(_: UserCreated) -> None:
            raise RuntimeError("boom")

        async def healthy_handler(event: UserCreated) -> None:
            received.append(event)

        bus.on(UserCreated, broken_handler)
        bus.on(UserCreated, healthy_handler)
        handle = await bus.dispatch(UserCreated(name="Mathis"))
        results = await handle.results()

        assert len(received) == 1
        assert len(results) == 2
        assert isinstance(results[0].exception, RuntimeError)
        assert results[1].ok

    @pytest.mark.asyncio
    async def test_nested_dispatch_completes_before_parent_handler_returns(
        self, bus: EventBus
    ) -> None:
        calls: list[str] = []

        async def on_user(_: UserCreated) -> None:
            calls.append("user:start")
            await bus.dispatch(OrderPlaced(order_id=1))
            calls.append("user:end")

        async def on_order(_: OrderPlaced) -> None:
            calls.append("order")

        bus.on(UserCreated, on_user)
        bus.on(OrderPlaced, on_order)
        await bus.dispatch(UserCreated(name="Mathis"))

        assert calls == ["user:start", "order", "user:end"]
