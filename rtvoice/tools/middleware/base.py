from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rtvoice.tools.di import ToolContext
from rtvoice.tools.results import ActionResult
from rtvoice.tools.views import Tool


@dataclass
class ToolCall:
    tool: Tool
    params: Any | None
    raw_args: dict[str, Any]
    context: ToolContext | None


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
