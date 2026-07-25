from collections.abc import Mapping, Sequence

from rtvoice.tools.middleware.base import ToolHandler, ToolMiddleware, compose
from rtvoice.tools.middleware.errors import ErrorBoundaryMiddleware
from rtvoice.tools.middleware.logging import CallLoggingMiddleware
from rtvoice.tools.middleware.resolution import ToolResolutionMiddleware
from rtvoice.tools.middleware.validation import ParamValidationMiddleware
from rtvoice.tools.views import Tool


class MiddlewareChain:
    """Assembles the tool pipeline with error boundary, resolution and
    validation always outermost, in that order.

    The ErrorBoundary comes first so every error — from any inner middleware or
    the tool itself — is turned into a failed ActionResult. Resolution and
    validation follow so that inner middlewares and the tool always see an
    available tool with validated params. Owning that order here keeps both
    guarantees out of call-site ordering.
    """

    def __init__(
        self,
        tools: Mapping[str, Tool],
        inner: Sequence[ToolMiddleware] | None = None,
    ) -> None:
        self._tools = tools
        self._inner: tuple[ToolMiddleware, ...] = (
            (CallLoggingMiddleware(),) if inner is None else tuple(inner)
        )

    def build(self, handler: ToolHandler) -> ToolHandler:
        return compose(
            (
                ErrorBoundaryMiddleware(),
                ToolResolutionMiddleware(self._tools),
                ParamValidationMiddleware(),
                *self._inner,
            ),
            handler,
        )
