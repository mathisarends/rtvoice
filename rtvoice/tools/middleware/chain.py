from collections.abc import Sequence

from rtvoice.tools.middleware.base import ToolHandler, ToolMiddleware, compose
from rtvoice.tools.middleware.defaults import default_tool_middlewares
from rtvoice.tools.middleware.errors import ErrorBoundaryMiddleware


class MiddlewareChain:
    """Assembles the tool pipeline with the ErrorBoundary always outermost.

    Placing the ErrorBoundary at the outermost position is what guarantees that
    every error — from any inner middleware or the tool itself — is caught and
    turned into a failed ActionResult. Owning that here keeps the guarantee out
    of call-site ordering.
    """

    def __init__(self, inner: Sequence[ToolMiddleware] | None = None) -> None:
        self._inner: tuple[ToolMiddleware, ...] = tuple(
            default_tool_middlewares() if inner is None else inner
        )

    def build(self, handler: ToolHandler) -> ToolHandler:
        return compose((ErrorBoundaryMiddleware(), *self._inner), handler)
