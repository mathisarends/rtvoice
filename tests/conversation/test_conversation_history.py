import pytest

from rtvoice.conversation import ConversationHistory
from rtvoice.events import EventBus
from rtvoice.events.views import (
    AssistantTranscriptCompletedEvent,
    UserTranscriptCompletedEvent,
)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def history(bus: EventBus) -> ConversationHistory:
    return ConversationHistory(bus)


class TestSubscription:
    @pytest.mark.asyncio
    async def test_appends_user_turn_on_user_event(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            UserTranscriptCompletedEvent(transcript="Hello", item_id="item-1")
        )

        assert len(history.turns) == 1
        assert history.turns[0].role == "user"
        assert history.turns[0].transcript == "Hello"

    @pytest.mark.asyncio
    async def test_appends_assistant_turn_on_assistant_event(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="Hi there", item_id="item-1", output_index=0, content_index=0
            )
        )

        assert len(history.turns) == 1
        assert history.turns[0].role == "assistant"
        assert history.turns[0].transcript == "Hi there"

    @pytest.mark.asyncio
    async def test_preserves_order_of_turns(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            UserTranscriptCompletedEvent(transcript="First", item_id="item-1")
        )
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="Second", item_id="item-2", output_index=0, content_index=0
            )
        )
        await bus.dispatch(
            UserTranscriptCompletedEvent(transcript="Third", item_id="item-3")
        )

        roles = [t.role for t in history.turns]
        assert roles == ["user", "assistant", "user"]


class TestTurns:
    @pytest.mark.asyncio
    async def test_turns_returns_copy(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            UserTranscriptCompletedEvent(transcript="Hello", item_id="item-1")
        )

        turns = history.turns
        turns.clear()

        assert len(history.turns) == 1

    def test_turns_empty_initially(self, history: ConversationHistory) -> None:
        assert history.turns == []


class TestFormat:
    def test_format_returns_placeholder_when_empty(
        self, history: ConversationHistory
    ) -> None:
        assert history.format() == "(no conversation yet)"

    @pytest.mark.asyncio
    async def test_format_includes_role_and_transcript(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            UserTranscriptCompletedEvent(transcript="Hello", item_id="item-1")
        )

        assert "[USER]: Hello" in history.format()

    @pytest.mark.asyncio
    async def test_format_separates_turns_with_newline(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            UserTranscriptCompletedEvent(transcript="Hello", item_id="item-1")
        )
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="Hi", item_id="item-2", output_index=0, content_index=0
            )
        )

        lines = history.format().split("\n")
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_format_uppercases_role(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="Hi", item_id="item-1", output_index=0, content_index=0
            )
        )

        assert "[ASSISTANT]:" in history.format()
