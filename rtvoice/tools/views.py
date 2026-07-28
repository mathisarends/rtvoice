from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from rtvoice.realtime.schemas import FunctionParameters, FunctionTool
from rtvoice.tools.binding import ToolAvailability, ToolDescription
from rtvoice.tools.di import ToolContext
from rtvoice.tools.schemas import build as build_schema


class ActionKind(StrEnum):
    GENERIC = "generic"
    READ = "read"
    MUTATE = "mutate"
    DESTRUCTIVE = "destructive"
    END_SESSION = "end_session"


def _make_dummy(param_model: type[BaseModel]) -> BaseModel:
    defaults: dict[str, Any] = {}
    for field_name, field in param_model.model_fields.items():
        annotation = field.annotation
        if annotation is str:
            defaults[field_name] = "placeholder"
        elif annotation is int:
            defaults[field_name] = 0
        elif annotation is float:
            defaults[field_name] = 0.0
        elif annotation is bool:
            defaults[field_name] = False
        else:
            defaults[field_name] = None
    return param_model.model_construct(**defaults)


class Tool:
    def __init__(
        self,
        name: str,
        description: str | ToolDescription,
        fn: Callable,
        *,
        param_model: type[BaseModel] | None = None,
        schema: FunctionParameters | None = None,
        result_instruction: str | None = None,
        respond: bool = True,
        status: str | Callable | None = None,
        kind: ActionKind = ActionKind.GENERIC,
        available_when: ToolAvailability | None = None,
    ):
        self.name = name
        self.description = description
        self.fn = fn
        self.param_model = param_model
        self.schema = schema or build_schema(fn, param_model=param_model)
        self.result_instruction = result_instruction
        self.respond = respond
        self.status = status
        self.kind = kind
        self.available_when = available_when
        self._validate_status()

    def is_available(self, context: ToolContext | None) -> bool:
        if self.available_when is None:
            return True
        return self.available_when(context)

    def resolve_description(self, context: ToolContext | None) -> str:
        if isinstance(self.description, ToolDescription):
            return self.description.resolve(context)
        return self.description

    def to_schema(self, context: ToolContext | None = None) -> FunctionTool:
        return FunctionTool(
            name=self.name,
            description=self.resolve_description(context),
            parameters=self.schema,
        )

    async def execute(self, arguments: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**arguments)
        return self.fn(**arguments)

    def format_status(self, args: BaseModel | dict[str, Any]) -> str | None:
        if self.status is None:
            return None

        if callable(self.status):
            if isinstance(args, BaseModel):
                return self.status(args)
            if self.param_model is not None:
                return self.status(self.param_model(**args))
            return None

        args_dict = (
            args.model_dump(exclude_none=True) if isinstance(args, BaseModel) else args
        )

        try:
            return self.status.format(**args_dict)
        except KeyError:
            return self.status

    def _validate_status(self) -> None:
        if self.status is None:
            return

        if self.param_model is None:
            raise ValueError(f"Tool '{self.name}': status requires a param_model")

        if not callable(self.status):
            self._validate_status_template(self.status)
            return

        dummy = _make_dummy(self.param_model)
        try:
            result = self.status(dummy)
            if not isinstance(result, str):
                raise ValueError(
                    f"Tool '{self.name}': status callable must return str, "
                    f"got {type(result).__name__}"
                )
        except ValueError:
            raise
        except AttributeError as exc:
            raise ValueError(
                f"Tool '{self.name}': status callable accesses a field that does not exist on "
                f"{self.param_model.__name__}: {exc}"
            ) from exc
        except Exception:
            # a status callable that trips over placeholder dummy values at
            # registration time is tolerated; it only needs to work at runtime
            pass

    def _validate_status_template(self, status: str) -> None:
        placeholders = {match.group(1) for match in re.finditer(r"\{(\w+)\}", status)}
        if not placeholders:
            return

        if self.param_model is None:
            raise ValueError(f"Tool '{self.name}': status requires a param_model")

        available_names = set(self.param_model.model_fields.keys())
        unknown_placeholders = placeholders - available_names
        if unknown_placeholders:
            raise ValueError(
                "Status template contains unknown placeholders: "
                f"{unknown_placeholders}. Available parameters: {available_names}"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tool):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)
