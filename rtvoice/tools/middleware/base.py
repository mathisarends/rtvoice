from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rtvoice.tools.di import ToolContext
from rtvoice.tools.results import ActionResult
from rtvoice.tools.views import Tool


@dataclass(frozen=True)
class ToolCall:
    name: str
    raw_args: dict[str, Any]
    context: ToolContext | None = None
    # filled in by the resolution and validation middlewares; everything inner
    # to them — including the tool itself — can rely on both being set
    tool: Tool | None = None
    params: Any | None = None


type ToolHandler = Callable[[ToolCall], Awaitable[ActionResult]]


class ToolMiddleware:
    async def __call__(self, call: ToolCall, next: ToolHandler) -> ActionResult:
        raise NotImplementedError


def compose(middlewares: Sequence[ToolMiddleware], handler: ToolHandler) -> ToolHandler:
    for middleware in reversed(middlewares):
        handler = _wrap(middleware, handler)
    return handler


def _wrap(middleware: ToolMiddleware, next_handler: ToolHandler) -> ToolHandler:
    async def handler(call: ToolCall) -> ActionResult:
        return await middleware(call, next_handler)

    return handler
