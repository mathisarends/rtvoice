from collections.abc import Mapping
from dataclasses import replace

from rtvoice.tools.di import ToolContext
from rtvoice.tools.middleware.base import ToolCall, ToolHandler, ToolMiddleware
from rtvoice.tools.results import ActionResult
from rtvoice.tools.views import Tool


class ToolResolutionMiddleware(ToolMiddleware):
    def __init__(self, tools: Mapping[str, Tool]) -> None:
        self._tools = tools

    async def __call__(self, call: ToolCall, next: ToolHandler) -> ActionResult:
        tool = self._tools.get(call.name)
        if tool is None or not tool.is_available(call.context):
            available = self._available_names(call.context)
            return ActionResult.fail(
                f"Unknown tool '{call.name}'. Available: {available}"
            )
        return await next(replace(call, tool=tool))

    def _available_names(self, context: ToolContext | None) -> list[str]:
        return [
            name for name, tool in self._tools.items() if tool.is_available(context)
        ]
