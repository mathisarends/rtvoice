import asyncio
from unittest.mock import AsyncMock

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.events.views import UserTranscriptCompletedEvent
from rtvoice.realtime.schemas import ConversationResponseCreateEvent
from rtvoice.supervisor.views import SupervisorAgentClarificationNeeded
from rtvoice.watchdogs import SupervisorInteractionWatchdog


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def websocket() -> AsyncMock:
    ws = AsyncMock()
    ws.send = AsyncMock()
    return ws


@pytest.fixture
def watchdog(
    event_bus: EventBus, websocket: AsyncMock
) -> SupervisorInteractionWatchdog:
    return SupervisorInteractionWatchdog(event_bus, websocket)


class TestClarificationNeeded:
    @pytest.mark.asyncio
    async def test_clarification_needed_sends_conversation_response(
        self,
        event_bus: EventBus,
        watchdog: SupervisorInteractionWatchdog,
        websocket: AsyncMock,
    ) -> None:
        clarification = SupervisorAgentClarificationNeeded(
            question="What do you mean?",
            answer_future=asyncio.get_event_loop().create_future(),
        )

        await event_bus.dispatch(clarification)

        websocket.send.assert_called_once()
        sent = websocket.send.call_args[0][0]
        assert isinstance(sent, ConversationResponseCreateEvent)

    @pytest.mark.asyncio
    async def test_clarification_stores_pending(
        self,
        event_bus: EventBus,
        watchdog: SupervisorInteractionWatchdog,
    ) -> None:
        clarification = SupervisorAgentClarificationNeeded(
            question="Are you sure?",
            answer_future=asyncio.get_event_loop().create_future(),
        )

        await event_bus.dispatch(clarification)

        assert watchdog._pending_clarification is clarification

    @pytest.mark.asyncio
    async def test_response_instructions_contain_question(
        self,
        event_bus: EventBus,
        watchdog: SupervisorInteractionWatchdog,
        websocket: AsyncMock,
    ) -> None:
        clarification = SupervisorAgentClarificationNeeded(
            question="Which city do you mean?",
            answer_future=asyncio.get_event_loop().create_future(),
        )

        await event_bus.dispatch(clarification)

        sent: ConversationResponseCreateEvent = websocket.send.call_args[0][0]
        assert "Which city do you mean?" in sent.response.instructions


class TestUserTranscriptResponse:
    @pytest.mark.asyncio
    async def test_user_transcript_without_pending_is_ignored(
        self,
        event_bus: EventBus,
        watchdog: SupervisorInteractionWatchdog,
    ) -> None:
        await event_bus.dispatch(
            UserTranscriptCompletedEvent(transcript="Yes", item_id="item_001")
        )

        assert watchdog._pending_clarification is None

    @pytest.mark.asyncio
    async def test_user_transcript_resolves_answer_future(
        self,
        event_bus: EventBus,
        watchdog: SupervisorInteractionWatchdog,
    ) -> None:
        clarification = SupervisorAgentClarificationNeeded(
            question="Do you confirm?",
            answer_future=asyncio.get_event_loop().create_future(),
        )
        await event_bus.dispatch(clarification)

        await event_bus.dispatch(
            UserTranscriptCompletedEvent(
                transcript="Yes, confirmed.", item_id="item_002"
            )
        )

        assert clarification.answer_future.result() == "Yes, confirmed."

    @pytest.mark.asyncio
    async def test_pending_cleared_after_user_transcript(
        self,
        event_bus: EventBus,
        watchdog: SupervisorInteractionWatchdog,
    ) -> None:
        clarification = SupervisorAgentClarificationNeeded(
            question="Which option?",
            answer_future=asyncio.get_event_loop().create_future(),
        )
        await event_bus.dispatch(clarification)
        await event_bus.dispatch(
            UserTranscriptCompletedEvent(transcript="Option A.", item_id="item_003")
        )

        assert watchdog._pending_clarification is None
