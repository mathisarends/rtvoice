import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import suppress

from pydantic import BaseModel, ValidationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from rtvoice.realtime.port import RealtimeProvider
from rtvoice.realtime.schemas import ServerEventAdapter
from rtvoice.views import RealtimeModel

logger = logging.getLogger(__name__)


class RealtimeWebSocket:
    def __init__(
        self,
        model: RealtimeModel,
        provider: RealtimeProvider,
    ):
        self._model = model
        self._provider = provider

        self._ws: ClientConnection | None = None
        self._receive_task: asyncio.Task | None = None
        self._is_connected: bool = False
        self._event_queue: asyncio.Queue = asyncio.Queue()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def connect(self) -> None:
        if self._ws:
            logger.debug("Closing existing connection")
            await self.close()

        url = self._provider.build_url(self._model.value)
        headers = self._provider.build_headers()

        logger.info("Connecting to %s...", url)

        try:
            self._ws = await connect(url, additional_headers=headers)
            self._is_connected = True
            logger.info("Connected successfully")
            self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            self._is_connected = False
            logger.error("Connection failed: %s", e)
            raise

    async def send(self, message: BaseModel) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")

        payload = message.model_dump(exclude_none=True)
        await self._ws.send(json.dumps(payload))

    async def close(self) -> None:
        if not self._ws:
            return

        logger.info("Closing connection...")
        self._is_connected = False

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._receive_task

        await self._ws.close()
        self._ws = None
        logger.info("Connection closed")

    async def events(self) -> AsyncGenerator:
        while True:
            event = await self._event_queue.get()
            if event is None:
                return
            yield event

    async def _receive_loop(self) -> None:
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    event = ServerEventAdapter.validate_python(data)
                    self._event_queue.put_nowait(event)
                except ValidationError:
                    logger.debug(
                        "Skipping unknown event type: %s",
                        data.get("type", "unknown"),
                    )
        except ConnectionClosed as e:
            self._is_connected = False
            logger.info("Connection closed: %s", e)
        finally:
            self._event_queue.put_nowait(None)
