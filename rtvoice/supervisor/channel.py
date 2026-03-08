from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class StatusMessage:
    message: str


@dataclass
class UserQuestion:
    question: str
    answer_future: asyncio.Future[str]


type SupervisorChannelEvent = StatusMessage | UserQuestion


class SupervisorChannel:
    def __init__(self, min_status_interval: float = 8.0) -> None:
        self._queue: asyncio.Queue[SupervisorChannelEvent] = asyncio.Queue()
        self._cancel_event = asyncio.Event()
        self._close_event = asyncio.Event()
        self._min_status_interval = min_status_interval
        self._last_status_at: float = 0.0
        self._pending_statuses: list[str] = []

    async def send_status(self, message: str) -> None:
        if self._close_event.is_set():
            return

        now = time.monotonic()
        self._pending_statuses.append(message)

        if now - self._last_status_at < self._min_status_interval:
            return

        self._last_status_at = now
        bundled = self._flush_pending()
        await self._queue.put(StatusMessage(message=bundled))

    def _flush_pending(self) -> str:
        messages = self._pending_statuses.copy()
        self._pending_statuses.clear()
        if len(messages) == 1:
            return messages[0]
        return " → ".join(messages)

    async def ask_user(self, question: str) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        await self._queue.put(UserQuestion(question=question, answer_future=future))
        return await future

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self) -> None:
        self._cancel_event.set()

    def close(self) -> None:
        # Flush buffered statuses before closing
        if self._pending_statuses:
            bundled = self._flush_pending()
            self._queue.put_nowait(StatusMessage(message=bundled))
        self._close_event.set()

    async def events(self) -> AsyncIterator[SupervisorChannelEvent]:
        while True:
            get_task: asyncio.Task = asyncio.ensure_future(self._queue.get())
            close_task: asyncio.Task = asyncio.ensure_future(self._close_event.wait())

            try:
                done, pending = await asyncio.wait(
                    {get_task, close_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except BaseException:
                get_task.cancel()
                close_task.cancel()
                raise

            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

            if close_task in done and not get_task.done():
                while not self._queue.empty():
                    yield self._queue.get_nowait()
                return

            if get_task.done():
                try:
                    yield get_task.result()
                except Exception:
                    return
