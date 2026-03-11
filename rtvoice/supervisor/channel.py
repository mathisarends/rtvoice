import asyncio
import contextlib
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
    def __init__(self, post_speech_delay: float = 5.5) -> None:
        self._queue: asyncio.Queue[SupervisorChannelEvent] = asyncio.Queue()
        self._cancel_event = asyncio.Event()
        self._close_event = asyncio.Event()
        self._post_speech_delay = post_speech_delay
        self._pending_statuses: list[str] = []
        self._flush_task: asyncio.Task[None] | None = None

    async def send_status(self, message: str) -> None:
        """Buffer a status message. It will be flushed after the assistant
        stops speaking and post_speech_delay seconds have elapsed."""
        if self._close_event.is_set():
            return
        self._pending_statuses.append(message)

    def notify_assistant_stopped(self) -> None:
        """Signal that the assistant finished speaking.
        Starts (or restarts) the post-speech delay timer."""
        if self._close_event.is_set():
            return
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = asyncio.create_task(self._flush_after_delay())

    async def _flush_after_delay(self) -> None:
        try:
            await asyncio.sleep(self._post_speech_delay)
        except asyncio.CancelledError:
            return
        if self._pending_statuses and not self._close_event.is_set():
            bundled = self._flush_pending()
            await self._queue.put(StatusMessage(message=bundled))

    def _flush_pending(self) -> str:
        messages = self._pending_statuses.copy()
        self._pending_statuses.clear()
        return messages[0] if len(messages) == 1 else " → ".join(messages)

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
        self._pending_statuses.clear()
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._close_event.set()

    async def events(self) -> AsyncIterator[SupervisorChannelEvent]:
        while True:
            get_task = asyncio.ensure_future(self._queue.get())
            close_task = asyncio.ensure_future(self._close_event.wait())

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

            if close_task in done:
                return

            if get_task in done:
                try:
                    yield get_task.result()
                except Exception:
                    return
