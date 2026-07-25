import logging
import time

from rtvoice.tools.middleware.base import ToolCall, ToolHandler, ToolMiddleware
from rtvoice.tools.results import ActionResult

logger = logging.getLogger(__name__)


class CallLoggingMiddleware(ToolMiddleware):
    async def __call__(self, call: ToolCall, next: ToolHandler) -> ActionResult:
        logger.info(
            "[tool] %s called with arguments: %r", call.tool.name, call.raw_args
        )
        start = time.perf_counter()
        result = await next(call)
        elapsed_ms = (time.perf_counter() - start) * 1000
        status = "ok" if result.ok else "fail"
        logger.info("[tool] %s -> %s (%.0f ms)", call.tool.name, status, elapsed_ms)
        return result
