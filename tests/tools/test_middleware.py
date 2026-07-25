from rtvoice.tools.middleware import (
    CallLoggingMiddleware,
    ErrorBoundaryMiddleware,
    ParamValidationMiddleware,
    ToolResolutionMiddleware,
)
from rtvoice.tools.middleware.implementations import (
    CallLoggingMiddleware as ImplementationCallLoggingMiddleware,
)
from rtvoice.tools.middleware.implementations import (
    ErrorBoundaryMiddleware as ImplementationErrorBoundaryMiddleware,
)
from rtvoice.tools.middleware.implementations import (
    ParamValidationMiddleware as ImplementationParamValidationMiddleware,
)
from rtvoice.tools.middleware.implementations import (
    ToolResolutionMiddleware as ImplementationToolResolutionMiddleware,
)


def test_reexports_middleware_implementations() -> None:
    assert CallLoggingMiddleware is ImplementationCallLoggingMiddleware
    assert ErrorBoundaryMiddleware is ImplementationErrorBoundaryMiddleware
    assert ParamValidationMiddleware is ImplementationParamValidationMiddleware
    assert ToolResolutionMiddleware is ImplementationToolResolutionMiddleware
