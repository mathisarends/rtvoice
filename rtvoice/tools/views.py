import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from rtvoice.realtime.schemas import FunctionParameters, FunctionTool


class VoidResult:
    def __str__(self) -> str:
        return "OK"


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
        description: str,
        function: Callable,
        schema: FunctionParameters,
        param_model: type[BaseModel] | None = None,
        result_instruction: str | None = None,
        holding_instruction: str | None = None,
        status: str | Callable | None = None,
        steering: str | None = None,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.schema = schema
        self.param_model = param_model
        self.result_instruction = result_instruction
        self.holding_instruction = holding_instruction
        self.status = status
        self.steering = steering
        self._validate_status()

    def _validate_status(self) -> None:
        if not callable(self.status):
            return
        if self.param_model is None:
            raise ValueError(
                f"Tool '{self.name}': callable status requires a param_model"
            )
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
            pass

    async def execute(self, arguments: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(self.function):
            result = await self.function(**arguments)
        else:
            result = self.function(**arguments)

        return result if result is not None else VoidResult()

    def to_pydantic(self) -> FunctionTool:
        return FunctionTool(
            name=self.name,
            description=self.description,
            parameters=self.schema,
        )

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tool):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)
