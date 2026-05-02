import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from websockets import frames
from websockets.exceptions import ConnectionClosed

from rtvoice.agent.views import RealtimeModel
from rtvoice.realtime.providers import OpenAIProvider
from rtvoice.realtime.websocket import RealtimeWebSocket


class SampleMessage(BaseModel):
    type: str
    value: str | None = None


def make_ws(messages: list[str] | None = None) -> MagicMock:
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()

    async def aiter_messages():
        for msg in messages or []:
            yield msg

    ws.__aiter__ = lambda self: aiter_messages()
    return ws


@pytest.fixture
def socket() -> RealtimeWebSocket:
    return RealtimeWebSocket(
        model=RealtimeModel.GPT_REALTIME,
        provider=OpenAIProvider(api_key="test-key"),
    )


class TestInit:
    def test_uses_provided_api_key(self) -> None:
        provider = OpenAIProvider(api_key="my-key")
        ws = RealtimeWebSocket(
            model=RealtimeModel.GPT_REALTIME,
            provider=provider,
        )

        assert ws._provider is provider

    def test_reads_api_key_from_env(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            provider = OpenAIProvider()
            _ = RealtimeWebSocket(
                model=RealtimeModel.GPT_REALTIME,
                provider=provider,
            )

        assert provider._api_key == "env-key"

    def test_raises_when_api_key_missing(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="OPENAI_API_KEY"),
        ):
            OpenAIProvider()

    def test_is_not_connected_initially(self, socket: RealtimeWebSocket) -> None:
        assert socket.is_connected is False


class TestConnect:
    @pytest.mark.asyncio
    async def test_sets_is_connected_true(self, socket: RealtimeWebSocket) -> None:
        ws = make_ws()

        with patch("rtvoice.realtime.websocket.connect", AsyncMock(return_value=ws)):
            await socket.connect()

        assert socket.is_connected is True

    @pytest.mark.asyncio
    async def test_starts_receive_task(self, socket: RealtimeWebSocket) -> None:
        ws = make_ws()

        with patch("rtvoice.realtime.websocket.connect", AsyncMock(return_value=ws)):
            await socket.connect()

        assert socket._receive_task is not None
        socket._receive_task.cancel()

    @pytest.mark.asyncio
    async def test_closes_existing_connection_before_reconnecting(
        self, socket: RealtimeWebSocket
    ) -> None:
        first_ws = make_ws()
        second_ws = make_ws()

        with patch(
            "rtvoice.realtime.websocket.connect",
            AsyncMock(side_effect=[first_ws, second_ws]),
        ):
            await socket.connect()
            await socket.connect()

        first_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_sets_is_connected_false_on_failure(
        self, socket: RealtimeWebSocket
    ) -> None:
        with (
            patch(
                "rtvoice.realtime.websocket.connect",
                AsyncMock(side_effect=OSError("refused")),
            ),
            pytest.raises(OSError),
        ):
            await socket.connect()

        assert socket.is_connected is False

    @pytest.mark.asyncio
    async def test_connects_with_bearer_token_in_headers(
        self, socket: RealtimeWebSocket
    ) -> None:
        ws = make_ws()
        captured = {}

        async def fake_connect(url, additional_headers=None, **kwargs):
            captured["headers"] = additional_headers
            return ws

        with patch("rtvoice.realtime.websocket.connect", fake_connect):
            await socket.connect()

        assert captured["headers"]["Authorization"] == "Bearer test-key"
        socket._receive_task.cancel()


class TestSend:
    @pytest.mark.asyncio
    async def test_sends_serialized_message(self, socket: RealtimeWebSocket) -> None:
        ws = make_ws()

        with patch("rtvoice.realtime.websocket.connect", AsyncMock(return_value=ws)):
            await socket.connect()
            await socket.send(SampleMessage(type="ping"))

        ws.send.assert_called_once()
        payload = json.loads(ws.send.call_args[0][0])
        assert payload["type"] == "ping"
        socket._receive_task.cancel()

    @pytest.mark.asyncio
    async def test_excludes_none_fields_from_payload(
        self, socket: RealtimeWebSocket
    ) -> None:
        ws = make_ws()

        with patch("rtvoice.realtime.websocket.connect", AsyncMock(return_value=ws)):
            await socket.connect()
            await socket.send(SampleMessage(type="ping", value=None))

        payload = json.loads(ws.send.call_args[0][0])
        assert "value" not in payload
        socket._receive_task.cancel()

    @pytest.mark.asyncio
    async def test_raises_when_not_connected(self, socket: RealtimeWebSocket) -> None:
        with pytest.raises(RuntimeError, match="Not connected"):
            await socket.send(SampleMessage(type="ping"))


class TestClose:
    @pytest.mark.asyncio
    async def test_sets_is_connected_false(self, socket: RealtimeWebSocket) -> None:
        ws = make_ws()

        with patch("rtvoice.realtime.websocket.connect", AsyncMock(return_value=ws)):
            await socket.connect()
            await socket.close()

        assert socket.is_connected is False

    @pytest.mark.asyncio
    async def test_closes_websocket(self, socket: RealtimeWebSocket) -> None:
        ws = make_ws()

        with patch("rtvoice.realtime.websocket.connect", AsyncMock(return_value=ws)):
            await socket.connect()
            await socket.close()

        ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancels_receive_task(self, socket: RealtimeWebSocket) -> None:
        ws = make_ws()

        with patch("rtvoice.realtime.websocket.connect", AsyncMock(return_value=ws)):
            await socket.connect()
            task = socket._receive_task
            await socket.close()

        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_close_without_connect_does_not_raise(
        self, socket: RealtimeWebSocket
    ) -> None:
        await socket.close()

    @pytest.mark.asyncio
    async def test_sets_ws_to_none(self, socket: RealtimeWebSocket) -> None:
        ws = make_ws()

        with patch("rtvoice.realtime.websocket.connect", AsyncMock(return_value=ws)):
            await socket.connect()
            await socket.close()

        assert socket._ws is None


class TestReceiveLoop:
    @pytest.mark.asyncio
    async def test_dispatches_valid_events_to_bus(
        self, socket: RealtimeWebSocket
    ) -> None:
        valid_event = json.dumps(
            {"type": "session.created", "event_id": "evt_1", "session": {}}
        )
        ws = make_ws([valid_event])

        with patch("rtvoice.realtime.websocket.connect", AsyncMock(return_value=ws)):
            await socket.connect()
            event = await asyncio.wait_for(socket.events().__anext__(), timeout=0.2)

        assert event.type == "session.created"

    @pytest.mark.asyncio
    async def test_skips_unknown_event_types(self, socket: RealtimeWebSocket) -> None:
        unknown_event = json.dumps({"type": "totally.unknown.event"})
        ws = make_ws([unknown_event])

        with patch("rtvoice.realtime.websocket.connect", AsyncMock(return_value=ws)):
            await socket.connect()
            events = socket.events()
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(events.__anext__(), timeout=0.2)

    @pytest.mark.asyncio
    async def test_sets_is_connected_false_on_connection_closed(
        self, socket: RealtimeWebSocket
    ) -> None:
        ws = AsyncMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        close_frame = frames.Close(code=1000, reason="normal")

        async def aiter_raises():
            raise ConnectionClosed(close_frame, close_frame, rcvd_then_sent=True)
            yield  # make it a generator

        ws.__aiter__ = lambda self: aiter_raises()

        with patch("rtvoice.realtime.websocket.connect", AsyncMock(return_value=ws)):
            await socket.connect()
            await asyncio.sleep(0.05)

        assert socket.is_connected is False
