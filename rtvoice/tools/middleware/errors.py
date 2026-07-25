import logging

from rtvoice.tools.middleware.base import ToolCall, ToolHandler, ToolMiddleware
from rtvoice.tools.results import ActionResult

logger = logging.getLogger(__name__)


class ToolFeedbackError(Exception):
    """Expected tool failure whose message is safe to hand back to the model."""


class ErrorBoundaryMiddleware(ToolMiddleware):
    async def __call__(self, call: ToolCall, next: ToolHandler) -> ActionResult:
        try:
            return await next(call)
        except ToolFeedbackError as error:
            return ActionResult.fail(error)
        except Exception:
            logger.exception("Tool '%s' failed", call.tool.name)
            return ActionResult.fail("Internal tool error.")
