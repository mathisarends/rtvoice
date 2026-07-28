import pytest
from transitbus import EventBus

from rtvoice.conversation import AssistantTurn, ConversationHistory, UserTurn
from rtvoice.events.views import (
    AssistantInterruptedEvent,
    AssistantTranscriptCompletedEvent,
    ToolExecutedEvent,
    UserTranscriptCompletedEvent,
)
from rtvoice.tools import ActionKind


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


class TestSeed:
    def test_seed_prepends_turns(self, history: ConversationHistory) -> None:
        history.seed(
            [
                UserTurn(transcript="Seeded question"),
                AssistantTurn(transcript="Seeded answer"),
            ]
        )

        assert [t.role for t in history.turns] == ["user", "assistant"]
        assert history.turns[0].transcript == "Seeded question"

    @pytest.mark.asyncio
    async def test_seeded_turns_precede_live_turns(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        history.seed([UserTurn(transcript="Seeded")])
        await bus.dispatch(
            UserTranscriptCompletedEvent(transcript="Live", item_id="item-1")
        )

        assert [t.transcript for t in history.turns] == ["Seeded", "Live"]


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


class TestToolExecutedTurns:
    @pytest.mark.asyncio
    async def test_appends_tool_turn_on_tool_executed_event(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            ToolExecutedEvent(
                name="load_skill",
                action_kind=ActionKind.GENERIC,
                silent=True,
                result="<skill_content>...</skill_content>",
            )
        )

        assert len(history.turns) == 1
        assert history.turns[0].role == "tool"
        assert history.turns[0].transcript == (
            "load_skill: <skill_content>...</skill_content>"
        )

    @pytest.mark.asyncio
    async def test_format_includes_tool_turn(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            ToolExecutedEvent(
                name="turn_on_lights",
                action_kind=ActionKind.MUTATE,
                silent=True,
                result="OK",
            )
        )

        assert "[TOOL]: turn_on_lights: OK" in history.format()


class TestInterruptions:
    @pytest.mark.asyncio
    async def test_marks_completed_assistant_turn_by_item(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="This was not fully heard.",
                item_id="item-1",
                output_index=0,
                content_index=0,
            )
        )
        await bus.dispatch(AssistantInterruptedEvent(item_id="item-1", played_ms=500))

        assert history.turns[0].interrupted is True
        assert history.turns[0].played_ms == 500

    @pytest.mark.asyncio
    async def test_marks_assistant_turn_when_interrupt_arrives_first(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            AssistantInterruptedEvent(response_id="response-1", played_ms=500)
        )
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="This was not fully heard.",
                item_id="item-1",
                output_index=0,
                content_index=0,
                response_id="response-1",
            )
        )

        assert history.turns[0].interrupted is True
        assert history.turns[0].played_ms == 500

    @pytest.mark.asyncio
    async def test_does_not_mark_unrelated_assistant_turn(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="This was fully heard.",
                item_id="item-1",
                output_index=0,
                content_index=0,
            )
        )
        await bus.dispatch(AssistantInterruptedEvent(item_id="item-2", played_ms=500))

        assert history.turns[0].interrupted is False


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

    @pytest.mark.asyncio
    async def test_two_seconds_at_default_speed_keeps_five_words(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        transcript = "one two three four five six seven eight nine ten"
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript=transcript,
                item_id="item-1",
                output_index=0,
                content_index=0,
            )
        )
        await bus.dispatch(AssistantInterruptedEvent(item_id="item-1", played_ms=2_000))

        assert history.format() == (
            "[ASSISTANT, INTERRUPTED]: one two three four five <INTERRUPTED>"
        )

    @pytest.mark.asyncio
    async def test_speech_speed_scales_estimated_word_count(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="one two three four five six seven eight nine ten",
                item_id="item-1",
                output_index=0,
                content_index=0,
            )
        )
        await bus.dispatch(
            AssistantInterruptedEvent(
                item_id="item-1",
                played_ms=2_000,
                speech_speed=1.5,
            )
        )

        assert history.format() == (
            "[ASSISTANT, INTERRUPTED]: one two three four five six seven <INTERRUPTED>"
        )

    @pytest.mark.asyncio
    async def test_fractional_word_estimate_rounds_down(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="one two three",
                item_id="item-1",
                output_index=0,
                content_index=0,
            )
        )
        await bus.dispatch(AssistantInterruptedEvent(item_id="item-1", played_ms=799))

        assert history.format() == ("[ASSISTANT, INTERRUPTED]: one <INTERRUPTED>")

    @pytest.mark.asyncio
    async def test_less_than_one_estimated_word_keeps_only_marker(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="one two three",
                item_id="item-1",
                output_index=0,
                content_index=0,
            )
        )
        await bus.dispatch(AssistantInterruptedEvent(item_id="item-1", played_ms=399))

        assert history.format() == "[ASSISTANT, INTERRUPTED]: <INTERRUPTED>"

    @pytest.mark.asyncio
    async def test_missing_playback_duration_keeps_only_marker(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="one two three",
                item_id="item-1",
                output_index=0,
                content_index=0,
            )
        )
        await bus.dispatch(AssistantInterruptedEvent(item_id="item-1"))

        assert history.format() == "[ASSISTANT, INTERRUPTED]: <INTERRUPTED>"

    @pytest.mark.asyncio
    async def test_estimate_longer_than_transcript_keeps_full_text_and_marker(
        self, bus: EventBus, history: ConversationHistory
    ) -> None:
        await bus.dispatch(
            AssistantTranscriptCompletedEvent(
                transcript="one two three",
                item_id="item-1",
                output_index=0,
                content_index=0,
            )
        )
        await bus.dispatch(
            AssistantInterruptedEvent(item_id="item-1", played_ms=10_000)
        )

        assert history.format() == (
            "[ASSISTANT, INTERRUPTED]: one two three <INTERRUPTED>"
        )
