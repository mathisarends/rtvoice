from dataclasses import replace

from pydantic import ValidationError

from rtvoice.tools.middleware.base import ToolCall, ToolHandler, ToolMiddleware
from rtvoice.tools.results import ActionResult


class ParamValidationMiddleware(ToolMiddleware):
    async def __call__(self, call: ToolCall, next: ToolHandler) -> ActionResult:
        param_model = call.tool.param_model
        if param_model is None:
            return await next(call)

        try:
            params = param_model.model_validate(call.raw_args)
        except ValidationError as error:
            return ActionResult.fail(error)

        return await next(replace(call, params=params))
