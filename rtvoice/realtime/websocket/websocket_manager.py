import asyncio
import json
import os
from contextlib import suppress
from typing import Self

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from rtvoice.events import EventBus
from rtvoice.realtime.schemas import ServerEvent
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.views import RealtimeModel

load_dotenv(override=True)


class RealtimeWebSocket(LoggingMixin):
    _BASE_URL = "wss://api.openai.com/v1/realtime"

    def __init__(
        self,
        model: RealtimeModel,
        event_bus: EventBus,
        api_key: str | None = None,
    ):
        self._model = model
        self._event_bus = event_bus
        self._api_key = api_key or self._get_api_key_from_env()

        self._ws: ClientConnection | None = None
        self._receive_task: asyncio.Task | None = None
        self._is_connected: bool = False

    def _get_api_key_from_env(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        return api_key

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._ws:
            self.logger.debug("Closing existing connection")
            await self.close()

        url = f"{self._BASE_URL}?model={self._model.value}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        self.logger.info("Connecting to %s...", url)

        try:
            self._ws = await connect(url, additional_headers=headers)
            self._is_connected = True
            self.logger.info("Connected successfully")
            self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            self._is_connected = False
            self.logger.error("Connection failed: %s", e)
            raise

    async def send(self, message: BaseModel) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")

        payload = message.model_dump(exclude_none=True)
        await self._ws.send(json.dumps(payload))

    async def close(self) -> None:
        if not self._ws:
            return

        self.logger.info("Closing connection...")
        self._is_connected = False

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._receive_task

        await self._ws.close()
        self._ws = None
        self.logger.info("Connection closed")

    async def _receive_loop(self) -> None:
        try:
            async for message in self._ws:
                data = json.loads(message)

                event = ServerEvent.model_validate(data)
                await self._event_bus.dispatch(event)

        except ValidationError as e:
            self.logger.warning("Invalid event received: %s", e)
        except ConnectionClosed as e:
            self._is_connected = False
            self.logger.info("Connection closed: %s", e)
