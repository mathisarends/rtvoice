from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from transitbus import EventBus

from rtvoice.agent.views import (
    InjectedAssistantMessage,
    InjectedConversation,
    InjectedUserMessage,
    RealtimeModel,
)
from rtvoice.audio import AudioSession
from rtvoice.events.views import AgentSessionConnectedEvent
from rtvoice.realtime.schemas import ConversationItemCreateEvent, SessionUpdateEvent
from rtvoice.realtime.session import RealtimeSession
from rtvoice.realtime.session_settings import RealtimeSessionSettings
from rtvoice.tools import Tools


class FakeWebSocket:
    def __init__(self) -> None:
        self.is_connected = False
        self.connect = AsyncMock(side_effect=self._connect)
        self.close = AsyncMock(side_effect=self._close)
        self.send = AsyncMock()

    async def _connect(self) -> None:
        self.is_connected = True

    async def _close(self) -> None:
        self.is_connected = False

    async def events(self) -> AsyncGenerator[object]:
        if False:
            yield object()


def make_session(
    *,
    injected_conversation: InjectedConversation | None = None,
) -> tuple[RealtimeSession, FakeWebSocket, list[str]]:
    event_bus = EventBus()
    websocket = FakeWebSocket()
    call_order: list[str] = []

    async def record_connected(_: AgentSessionConnectedEvent) -> None:
        call_order.append("connected")

    async def record_send(event: object) -> None:
        call_order.append(type(event).__name__)

    websocket.send.side_effect = record_send
    event_bus.on(AgentSessionConnectedEvent, record_connected)

    with patch.object(RealtimeSession, "_setup_handlers"):
        session = RealtimeSession(
            event_bus=event_bus,
            settings=RealtimeSessionSettings(
                model=RealtimeModel.GPT_REALTIME_2_1_MINI,
                instructions="Test assistant",
            ),
            tools=Tools(),
            audio_session=MagicMock(spec=AudioSession),
            provider=MagicMock(),
            injected_conversation=injected_conversation,
        )
    session._websocket = websocket

    return session, websocket, call_order


class TestInjectedConversation:
    @pytest.mark.asyncio
    async def test_start_injects_conversation_after_session_update_before_connected_event(
        self,
    ) -> None:
        conversation = InjectedConversation(
            messages=[
                InjectedUserMessage(text="Mein Name ist Max."),
                InjectedAssistantMessage(text="Hallo Max, wie kann ich helfen?"),
            ]
        )
        session, websocket, call_order = make_session(
            injected_conversation=conversation
        )

        await session.start()
        await session.stop()

        assert call_order == [
            "SessionUpdateEvent",
            "ConversationItemCreateEvent",
            "ConversationItemCreateEvent",
            "connected",
        ]
        assert isinstance(websocket.send.call_args_list[0].args[0], SessionUpdateEvent)

    @pytest.mark.asyncio
    async def test_start_does_not_send_conversation_items_without_injected_conversation(
        self,
    ) -> None:
        session, websocket, _ = make_session()

        await session.start()
        await session.stop()

        sent_events = [call.args[0] for call in websocket.send.call_args_list]
        assert not any(
            isinstance(event, ConversationItemCreateEvent) for event in sent_events
        )

    def test_injected_user_message_uses_input_text_content(self) -> None:
        session, _, _ = make_session()
        event = session._injected_message_event(
            InjectedUserMessage(text="Ich bin Max.")
        )

        payload = event.model_dump(exclude_none=True)
        assert payload["item"]["role"] == "user"
        assert payload["item"]["content"] == [
            {"type": "input_text", "text": "Ich bin Max."}
        ]

    def test_injected_assistant_message_uses_output_text_content(self) -> None:
        session, _, _ = make_session()
        event = session._injected_message_event(
            InjectedAssistantMessage(text="Hallo Max.")
        )

        payload = event.model_dump(exclude_none=True)
        assert payload["item"]["role"] == "assistant"
        assert payload["item"]["content"] == [
            {"type": "output_text", "text": "Hallo Max."}
        ]


class TestSessionUpdate:
    @pytest.mark.asyncio
    async def test_session_update_carries_the_settings_and_tool_schema(self) -> None:
        session, websocket, _ = make_session()

        await session.start()
        await session.stop()

        sent = websocket.send.call_args_list[0].args[0]
        assert isinstance(sent, SessionUpdateEvent)
        assert sent.session.instructions == "Test assistant"
        assert sent.session.model == RealtimeModel.GPT_REALTIME_2_1_MINI
        assert sent.session.tools == session._tools.get_schema()
