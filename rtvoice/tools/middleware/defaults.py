from collections.abc import Sequence

from rtvoice.tools.middleware.base import ToolMiddleware
from rtvoice.tools.middleware.logging import CallLoggingMiddleware


def default_tool_middlewares() -> Sequence[ToolMiddleware]:
    return (CallLoggingMiddleware(),)
