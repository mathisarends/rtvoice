import asyncio
import json
import threading
from typing import Any, Self

import websocket
from pydantic import BaseModel

from rtvoice.config import AgentEnv
from rtvoice.events import EventBus, EventDispatcher
from rtvoice.events.schemas import RealtimeModel
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.state.base import VoiceAssistantEvent


class WebSocketManager(LoggingMixin):
    DEFAULT_BASE_URL = "wss://api.openai.com/v1/realtime"

    def __init__(
        self,
        websocket_url: str,
        headers: dict[str, str],
        event_bus: EventBus,
    ):
        self._websocket_url = websocket_url
        self._headers = [f"{k}: {v}" for k, v in headers.items()] if headers else []
        self._ws: websocket.WebSocketApp | None = None
        self._connected = False
        self._connection_event = threading.Event()
        self._running = False
        self._event_bus = event_bus
        self._event_dispatcher = EventDispatcher(self._event_bus)
        self._connection_lock = asyncio.Lock()

    @classmethod
    def from_model(
        cls,
        *,
        model: RealtimeModel = RealtimeModel.GPT_REALTIME,
        event_bus: EventBus,
        env: AgentEnv | None = None,
    ) -> Self:
        env = env or AgentEnv()
        ws_url = cls._get_websocket_url(model.value)
        headers = cls._get_auth_header(env.openai_api_key)
        return cls(ws_url, headers, event_bus)

    async def create_connection(self) -> None:
        async with self._connection_lock:
            self.logger.info("Establishing connection to %s...", self._websocket_url)

            if self._ws:
                self.logger.debug("Closing existing connection before creating new one")
                await self._close_internal()

            self._connection_event.clear()

            self._ws = websocket.WebSocketApp(
                self._websocket_url,
                header=self._headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            self._running = True
            ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
            ws_thread.start()

            connected = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._connection_event.wait(timeout=10)
            )

            if not (connected and self._connected):
                self._ws = None
                raise RuntimeError("Failed to establish connection within timeout")

    async def send_message(self, message: dict[str, Any] | BaseModel) -> None:
        if not self._connected or not self._ws:
            raise RuntimeError(
                "No connection available. Call create_connection() first."
            )

        payload = (
            message.model_dump(exclude_none=True)
            if isinstance(message, BaseModel)
            else message
        )

        await asyncio.get_event_loop().run_in_executor(
            None, self._send_message_sync, payload
        )

    async def close(self) -> None:
        async with self._connection_lock:
            await self._close_internal()

    async def _close_internal(self) -> None:
        if not self._ws:
            return

        self.logger.info("Closing connection...")

        self._running = False
        self._connected = False

        await asyncio.get_event_loop().run_in_executor(None, self._close_sync)
        self._ws = None
        self.logger.info("Connection closed")

    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    def _on_open(self, ws) -> None:
        self.logger.info("Connection successfully established!")
        self._connected = True
        self._connection_event.set()

    def _on_message(self, ws, message: str) -> None:
        data = json.loads(message)
        self._event_dispatcher.dispatch_event(data)

    def _on_error(self, ws, error) -> None:
        self.logger.error("WebSocket error: %s", error)
        self._connected = False
        self._event_bus.publish_sync(
            VoiceAssistantEvent.ERROR_OCCURRED, {"error": str(error)}
        )

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        self.logger.info("Connection closed: %s %s", close_status_code, close_msg)
        self._connected = False
        self._running = False

    def _send_message_sync(self, message: dict[str, Any]) -> None:
        self._ws.send(json.dumps(message))

    def _close_sync(self) -> None:
        if self._ws:
            self._ws.close()

    @classmethod
    def _get_websocket_url(cls, model: str) -> str:
        return f"{cls.DEFAULT_BASE_URL}?model={model}"

    @classmethod
    def _get_auth_header(
        cls,
        api_key: str,
    ) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}
