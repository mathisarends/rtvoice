import pytest

from rtvoice.events.bus import EventBus
from rtvoice.events.views import AgentErrorEvent
from rtvoice.realtime.schemas import ErrorDetails, ErrorEvent, RealtimeServerEvent
from rtvoice.watchdogs import ErrorWatchdog


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def watchdog(event_bus: EventBus) -> ErrorWatchdog:
    return ErrorWatchdog(event_bus)


class TestErrorMapping:
    @pytest.mark.asyncio
    async def test_error_event_dispatches_agent_error_event(
        self, event_bus: EventBus, watchdog: ErrorWatchdog
    ) -> None:
        received: list[AgentErrorEvent] = []

        async def capture(e: AgentErrorEvent) -> None:
            received.append(e)

        event_bus.subscribe(AgentErrorEvent, capture)
        await event_bus.dispatch(
            ErrorEvent(
                type=RealtimeServerEvent.ERROR,
                event_id="evt_001",
                error=ErrorDetails(
                    type="server_error",
                    message="Something went wrong",
                    event_id="evt_001",
                ),
            )
        )

        assert len(received) == 1
        assert received[0].error.type == "server_error"
        assert received[0].error.message == "Something went wrong"
        assert received[0].event_id == "evt_001"

    @pytest.mark.asyncio
    async def test_error_event_id_comes_from_error_details(
        self, event_bus: EventBus, watchdog: ErrorWatchdog
    ) -> None:
        received: list[AgentErrorEvent] = []

        async def capture(e: AgentErrorEvent) -> None:
            received.append(e)

        event_bus.subscribe(AgentErrorEvent, capture)
        await event_bus.dispatch(
            ErrorEvent(
                type=RealtimeServerEvent.ERROR,
                event_id="evt_002",
                error=ErrorDetails(
                    type="invalid_request",
                    message="Bad request",
                ),
            )
        )

        assert received[0].error.type == "invalid_request"
        assert received[0].error.message == "Bad request"
        assert received[0].event_id is None

    @pytest.mark.asyncio
    async def test_each_error_event_dispatches_exactly_one_agent_error(
        self, event_bus: EventBus, watchdog: ErrorWatchdog
    ) -> None:
        received: list[AgentErrorEvent] = []

        async def capture(e: AgentErrorEvent) -> None:
            received.append(e)

        event_bus.subscribe(AgentErrorEvent, capture)

        for i in range(3):
            await event_bus.dispatch(
                ErrorEvent(
                    type=RealtimeServerEvent.ERROR,
                    event_id=f"evt_{i:03d}",
                    error=ErrorDetails(type="server_error", message=f"Error {i}"),
                )
            )

        assert len(received) == 3
