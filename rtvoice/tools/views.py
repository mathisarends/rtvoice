import collections.abc
import inspect
import re
import types
from collections.abc import Callable
from enum import Enum
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from rtvoice.realtime.schemas import (
    FunctionParameterProperty,
    FunctionParameters,
    FunctionTool,
)
from rtvoice.tools.di import _INJECT_MARKER


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


class ToolSchemaBuilder:
    _PRIMITIVE_TYPES: ClassVar[dict[type, str]] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    _COLLECTION_TYPES: ClassVar[tuple[type, ...]] = (
        collections.abc.Sequence,
        collections.abc.Iterable,
        collections.abc.Collection,
    )

    def build(
        self, func: Callable, param_model: type[BaseModel] | None = None
    ) -> FunctionParameters:
        if param_model is not None:
            return self.build_from_model(param_model)

        signature = inspect.signature(func)
        type_hints = get_type_hints(func, include_extras=True)

        properties: dict[str, FunctionParameterProperty] = {}
        required_params: list[str] = []

        for param_name, param in signature.parameters.items():
            if self._should_skip_param(param_name, type_hints):
                continue

            param_type = type_hints.get(param_name, str)
            actual_type, description = self._extract_type_and_description(param_type)

            properties[param_name] = self._convert_to_json_schema(
                actual_type, description
            )

            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)

        return FunctionParameters(
            type="object",
            strict=True,
            properties=properties,
            required=required_params,
        )

    def build_from_model(self, model: type[BaseModel]) -> FunctionParameters:
        properties: dict[str, FunctionParameterProperty] = {}
        required_params: list[str] = []

        for field_name, field_info in model.model_fields.items():
            description = self._field_description(field_info)
            field_type = field_info.annotation
            prop = self._convert_to_json_schema(field_type, description)

            if (
                not field_info.is_required()
                and field_info.default is not None
                and field_info.default is not PydanticUndefined
            ):
                prop = prop.model_copy(update={"default": field_info.default})

            properties[field_name] = prop

            if field_info.is_required():
                required_params.append(field_name)

        return FunctionParameters(
            type="object",
            properties=properties,
            required=required_params,
        )

    def _should_skip_param(self, param_name: str, type_hints: dict[str, Any]) -> bool:
        if param_name in ("self", "cls"):
            return True

        param_type = type_hints.get(param_name)
        if not param_type:
            return False

        return self._has_inject_marker(param_type)

    def _has_inject_marker(self, type_hint: Any) -> bool:
        if get_origin(type_hint) is not Annotated:
            return False
        return any(isinstance(arg, type(_INJECT_MARKER)) for arg in get_args(type_hint))

    def _extract_type_and_description(self, type_hint: Any) -> tuple[Any, str | None]:
        if get_origin(type_hint) is not Annotated:
            return type_hint, None

        args = get_args(type_hint)
        description = next((arg for arg in args[1:] if isinstance(arg, str)), None)
        return args[0], description

    def _field_description(self, field_info: Any) -> str | None:
        if field_info.description is not None:
            return field_info.description

        return next(
            (metadata for metadata in field_info.metadata if isinstance(metadata, str)),
            None,
        )

    def _convert_to_json_schema(
        self, python_type: Any, description: str | None = None
    ) -> FunctionParameterProperty:
        origin = get_origin(python_type)

        if origin is Union or isinstance(python_type, types.UnionType):
            return self._handle_union_type(python_type, description)

        if origin is Literal:
            return FunctionParameterProperty(
                type="string",
                description=description,
                enum=[str(arg) for arg in get_args(python_type)],
            )

        if origin is list:
            return self._handle_list_type(python_type, description)

        if origin is dict:
            return FunctionParameterProperty(type="object", description=description)

        if origin in self._COLLECTION_TYPES:
            return FunctionParameterProperty(type="array", description=description)

        json_type = self._PRIMITIVE_TYPES.get(python_type)
        if json_type:
            return FunctionParameterProperty(type=json_type, description=description)

        if self._is_pydantic_model(python_type):
            return self._pydantic_to_schema(python_type, description)

        if self._is_enum(python_type):
            return self._enum_to_schema(python_type, description)

        return FunctionParameterProperty(type="string", description=description)

    def _handle_union_type(
        self, union_type: Any, description: str | None
    ) -> FunctionParameterProperty:
        non_none_args = [arg for arg in get_args(union_type) if arg is not type(None)]

        if len(non_none_args) == 1:
            return self._convert_to_json_schema(non_none_args[0], description)

        return FunctionParameterProperty(type="string", description=description)

    def _handle_list_type(
        self, list_type: Any, description: str | None
    ) -> FunctionParameterProperty:
        args = get_args(list_type)
        items_schema = self._convert_to_json_schema(args[0]) if args else None

        return FunctionParameterProperty(
            type="array",
            description=description,
            items=items_schema,
        )

    def _enum_to_schema(
        self, enum_type: type[Enum], description: str | None
    ) -> FunctionParameterProperty:
        return FunctionParameterProperty(
            type="string",
            description=description,
            enum=[str(item.value) for item in enum_type],
        )

    def _is_pydantic_model(self, python_type: Any) -> bool:
        try:
            return isinstance(python_type, type) and issubclass(python_type, BaseModel)
        except TypeError:
            return False

    def _is_enum(self, python_type: Any) -> bool:
        try:
            return isinstance(python_type, type) and issubclass(python_type, Enum)
        except TypeError:
            return False

    def _pydantic_to_schema(
        self, model: type[BaseModel], description: str | None = None
    ) -> FunctionParameterProperty:
        schema = self.build_from_model(model)
        return FunctionParameterProperty(
            type="object",
            description=description or model.__doc__,
            properties=schema.properties,
            required=schema.required or None,
        )


_schema_builder = ToolSchemaBuilder()


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        param_model: type[BaseModel] | None = None,
        schema: FunctionParameters | None = None,
        result_instruction: str | None = None,
        holding_instruction: str | None = None,
        status: str | Callable | None = None,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.param_model = param_model
        self.schema = schema or _schema_builder.build(function, param_model=param_model)
        self.result_instruction = result_instruction
        self.holding_instruction = holding_instruction
        self.status = status
        self._validate_status()

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
