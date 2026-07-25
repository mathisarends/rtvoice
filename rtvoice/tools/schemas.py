import collections.abc
import inspect
import types
from collections.abc import Callable
from enum import Enum
from typing import (
    Annotated,
    Any,
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
)
from rtvoice.tools.di import _INJECT_MARKER

_PRIMITIVE_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_COLLECTION_TYPES: tuple[type, ...] = (
    collections.abc.Sequence,
    collections.abc.Iterable,
    collections.abc.Collection,
)


def build(
    func: Callable | None = None,
    *,
    param_model: type[BaseModel] | None = None,
) -> FunctionParameters:
    if param_model is not None:
        return _build_from_model(param_model)
    if func is None:
        raise ValueError("build requires either a function or a param_model")

    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    properties: dict[str, FunctionParameterProperty] = {}
    required_params: list[str] = []

    for param_name, param in signature.parameters.items():
        if _should_skip_param(param_name, type_hints):
            continue

        param_type = type_hints.get(param_name, str)
        actual_type, description = _extract_type_and_description(param_type)
        properties[param_name] = _convert_to_json_schema(actual_type, description)

        if param.default == inspect.Parameter.empty:
            required_params.append(param_name)

    return FunctionParameters(
        type="object", properties=properties, required=required_params
    )


def _build_from_model(model: type[BaseModel]) -> FunctionParameters:
    properties: dict[str, FunctionParameterProperty] = {}
    required_params: list[str] = []

    for field_name, field_info in model.model_fields.items():
        description = _field_description(field_info)
        prop = _convert_to_json_schema(field_info.annotation, description)

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
        type="object", properties=properties, required=required_params
    )


def _should_skip_param(param_name: str, type_hints: dict[str, Any]) -> bool:
    if param_name in ("self", "cls"):
        return True

    param_type = type_hints.get(param_name)
    if not param_type:
        return False

    return _has_inject_marker(param_type)


def _has_inject_marker(type_hint: Any) -> bool:
    if get_origin(type_hint) is not Annotated:
        return False
    return any(isinstance(arg, type(_INJECT_MARKER)) for arg in get_args(type_hint))


def _extract_type_and_description(type_hint: Any) -> tuple[Any, str | None]:
    if get_origin(type_hint) is not Annotated:
        return type_hint, None

    args = get_args(type_hint)
    description = next((arg for arg in args[1:] if isinstance(arg, str)), None)
    return args[0], description


def _field_description(field_info: Any) -> str | None:
    if field_info.description is not None:
        return field_info.description

    return next(
        (metadata for metadata in field_info.metadata if isinstance(metadata, str)),
        None,
    )


def _convert_to_json_schema(
    python_type: Any, description: str | None = None
) -> FunctionParameterProperty:
    origin = get_origin(python_type)

    if origin is Union or isinstance(python_type, types.UnionType):
        return _handle_union_type(python_type, description)

    if origin is Literal:
        return FunctionParameterProperty(
            type="string",
            description=description,
            enum=[str(arg) for arg in get_args(python_type)],
        )

    if origin is list:
        return _handle_list_type(python_type, description)

    if origin is dict:
        return FunctionParameterProperty(type="object", description=description)

    if origin in _COLLECTION_TYPES:
        return FunctionParameterProperty(type="array", description=description)

    json_type = _PRIMITIVE_TYPES.get(python_type)
    if json_type:
        return FunctionParameterProperty(type=json_type, description=description)

    if _is_pydantic_model(python_type):
        return _pydantic_to_schema(python_type, description)

    if _is_enum(python_type):
        return _enum_to_schema(python_type, description)

    return FunctionParameterProperty(type="string", description=description)


def _handle_union_type(
    union_type: Any, description: str | None
) -> FunctionParameterProperty:
    non_none_args = [arg for arg in get_args(union_type) if arg is not type(None)]

    if len(non_none_args) == 1:
        return _convert_to_json_schema(non_none_args[0], description)

    return FunctionParameterProperty(type="string", description=description)


def _handle_list_type(
    list_type: Any, description: str | None
) -> FunctionParameterProperty:
    args = get_args(list_type)
    items_schema = _convert_to_json_schema(args[0]) if args else None

    return FunctionParameterProperty(
        type="array", description=description, items=items_schema
    )


def _enum_to_schema(
    enum_type: type[Enum], description: str | None
) -> FunctionParameterProperty:
    return FunctionParameterProperty(
        type="string",
        description=description,
        enum=[str(item.value) for item in enum_type],
    )


def _is_pydantic_model(python_type: Any) -> bool:
    try:
        return isinstance(python_type, type) and issubclass(python_type, BaseModel)
    except TypeError:
        return False


def _is_enum(python_type: Any) -> bool:
    try:
        return isinstance(python_type, type) and issubclass(python_type, Enum)
    except TypeError:
        return False


def _pydantic_to_schema(
    model: type[BaseModel], description: str | None = None
) -> FunctionParameterProperty:
    schema = _build_from_model(model)
    return FunctionParameterProperty(
        type="object",
        description=description or model.__doc__,
        properties=schema.properties,
        required=schema.required or None,
    )
