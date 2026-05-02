from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rtvoice.audio import AudioSession
from rtvoice.events import EventBus
from rtvoice.events.views import AgentSessionConnectedEvent
from rtvoice.realtime.schemas import ConversationItemCreateEvent, SessionUpdateEvent
from rtvoice.realtime.session import RealtimeSession
from rtvoice.tools import Tools
from rtvoice.views import (
    AssistantVoice,
    ConversationSeed,
    NoiseReduction,
    RealtimeModel,
    SeedMessage,
    SemanticVAD,
    TranscriptionModel,
)


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
    *, conversation_seed: ConversationSeed | None = None
) -> tuple[RealtimeSession, FakeWebSocket, list[str]]:
    event_bus = EventBus()
    websocket = FakeWebSocket()
    call_order: list[str] = []

    async def record_connected(_: AgentSessionConnectedEvent) -> None:
        call_order.append("connected")

    async def record_send(event: object) -> None:
        call_order.append(type(event).__name__)

    websocket.send.side_effect = record_send
    event_bus.subscribe(AgentSessionConnectedEvent, record_connected)

    with patch.object(RealtimeSession, "_setup_handlers"):
        session = RealtimeSession(
            event_bus=event_bus,
            model=RealtimeModel.GPT_REALTIME_MINI,
            instructions="Test assistant",
            voice=AssistantVoice.MARIN,
            speech_speed=1.0,
            transcription_model=TranscriptionModel.WHISPER_1,
            output_modalities=["audio"],
            noise_reduction=NoiseReduction.FAR_FIELD,
            turn_detection=SemanticVAD(),
            tools=Tools(),
            audio_session=MagicMock(spec=AudioSession),
            subagents=[],
            conversation_seed=conversation_seed,
            inactivity_timeout_enabled=False,
            inactivity_timeout_seconds=None,
            recording_path=None,
            provider=MagicMock(),
        )
    session._websocket = websocket

    return session, websocket, call_order


class TestConversationSeedInjection:
    @pytest.mark.asyncio
    async def test_start_injects_seed_after_session_update_before_connected_event(
        self,
    ) -> None:
        seed = ConversationSeed.from_pairs(
            ("Mein Name ist Max.", "Hallo Max, wie kann ich helfen?")
        )
        session, websocket, call_order = make_session(conversation_seed=seed)

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
    async def test_start_does_not_send_conversation_items_without_seed(self) -> None:
        session, websocket, _ = make_session()

        await session.start()
        await session.stop()

        sent_events = [call.args[0] for call in websocket.send.call_args_list]
        assert not any(
            isinstance(event, ConversationItemCreateEvent) for event in sent_events
        )

    def test_seed_user_message_uses_input_text_content(self) -> None:
        session, _, _ = make_session()
        event = session._seed_message_event(SeedMessage.user("Ich bin Max."))

        payload = event.model_dump(exclude_none=True)
        assert payload["item"]["role"] == "user"
        assert payload["item"]["content"] == [
            {"type": "input_text", "text": "Ich bin Max."}
        ]

    def test_seed_assistant_message_uses_output_text_content(self) -> None:
        session, _, _ = make_session()
        event = session._seed_message_event(SeedMessage.assistant("Hallo Max."))

        payload = event.model_dump(exclude_none=True)
        assert payload["item"]["role"] == "assistant"
        assert payload["item"]["content"] == [
            {"type": "output_text", "text": "Hallo Max."}
        ]
