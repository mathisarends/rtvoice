import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class StatusMessage:
    message: str


class SupervisorChannel:
    def __init__(self, post_speech_delay: float = 5.5) -> None:
        self._outbox: asyncio.Queue[StatusMessage] = asyncio.Queue()
        self._cancelled = asyncio.Event()
        self._closed = asyncio.Event()
        self._post_speech_delay = post_speech_delay
        self._buffered_statuses: list[str] = []
        self._speech_end_timer: asyncio.Task[None] | None = None

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def buffer_status(self, message: str) -> None:
        if self._closed.is_set():
            return
        self._buffered_statuses.append(message)

    def notify_speech_ended(self) -> None:
        if self._closed.is_set():
            return
        self._restart_flush_timer()

    def cancel(self) -> None:
        self._cancelled.set()

    def close(self) -> None:
        self._buffered_statuses.clear()
        self._cancel_flush_timer()
        self._closed.set()

    async def events(self) -> AsyncIterator[StatusMessage]:
        while True:
            next_message_task = asyncio.ensure_future(self._outbox.get())
            channel_closed_task = asyncio.ensure_future(self._closed.wait())

            try:
                done, pending = await asyncio.wait(
                    {next_message_task, channel_closed_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except BaseException:
                next_message_task.cancel()
                channel_closed_task.cancel()
                raise

            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

            if channel_closed_task in done:
                return

            if next_message_task in done:
                try:
                    yield next_message_task.result()
                except Exception:
                    return

    def _restart_flush_timer(self) -> None:
        self._cancel_flush_timer()
        self._speech_end_timer = asyncio.create_task(
            self._flush_buffered_statuses_after_delay()
        )

    def _cancel_flush_timer(self) -> None:
        if self._speech_end_timer and not self._speech_end_timer.done():
            self._speech_end_timer.cancel()

    async def _flush_buffered_statuses_after_delay(self) -> None:
        try:
            await asyncio.sleep(self._post_speech_delay)
        except asyncio.CancelledError:
            return

        if self._buffered_statuses and not self._closed.is_set():
            await self._outbox.put(
                StatusMessage(message=self._bundle_and_clear_statuses())
            )

    def _bundle_and_clear_statuses(self) -> str:
        messages = self._buffered_statuses.copy()
        self._buffered_statuses.clear()
        return messages[0] if len(messages) == 1 else " → ".join(messages)
