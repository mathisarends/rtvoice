from __future__ import annotations

import inspect
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from rtvoice.tools.di import ToolContext, _InjectMarker
from rtvoice.tools.views import Tool


def _is_injectable(hint: Any) -> bool:
    if get_origin(hint) is not Annotated:
        return False
    return any(isinstance(metadata, _InjectMarker) for metadata in get_args(hint))


class ArgumentResolver:
    def resolve(
        self,
        tool: Tool,
        args: dict[str, Any],
        params: BaseModel | None,
        context: ToolContext | None,
    ) -> dict[str, Any]:
        kwargs = self._resolve_non_injected_args(tool, args, params)
        hints = get_type_hints(tool.fn, include_extras=True)
        signature = inspect.signature(tool.fn)

        for param_name, param in signature.parameters.items():
            hint = hints.get(param_name)
            if hint is None or not _is_injectable(hint):
                continue

            actual_type = get_args(hint)[0]
            dependency = context.resolve(actual_type) if context is not None else None
            if dependency is None:
                if param.default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Missing injected dependency for parameter '{param_name}' "
                        f"of type '{actual_type.__name__}'"
                    )
                continue
            kwargs[param_name] = dependency

        return kwargs

    def _resolve_non_injected_args(
        self,
        tool: Tool,
        args: dict[str, Any],
        params: BaseModel | None,
    ) -> dict[str, Any]:
        if tool.param_model is None:
            return dict(args)

        if params is None:
            raise ValueError(f"Missing parsed params for tool '{tool.name}'")

        hints = get_type_hints(tool.fn, include_extras=True)
        signature = inspect.signature(tool.fn)
        target = self._find_param_model_parameter(signature, hints, tool.param_model)
        if target is None:
            raise ValueError(
                f"Tool '{tool.name}' uses params model '{tool.param_model.__name__}' "
                "but has no parameter that can receive it"
            )
        return {target: params}

    def _find_param_model_parameter(
        self,
        signature: inspect.Signature,
        hints: dict[str, Any],
        param_model: type[BaseModel],
    ) -> str | None:
        candidates: list[str] = []
        for param_name in signature.parameters:
            if param_name in ("self", "cls"):
                continue
            hint = hints.get(param_name)
            if hint is not None and _is_injectable(hint):
                continue
            candidates.append(param_name)
            if hint == param_model:
                return param_name

        if len(candidates) == 1:
            return candidates[0]
        return None
