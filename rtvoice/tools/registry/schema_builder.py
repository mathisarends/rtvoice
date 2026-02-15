import collections.abc
import inspect
from collections.abc import Callable
from typing import Annotated, Any, ClassVar, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from rtvoice.realtime.schemas import FunctionParameterProperty, FunctionParameters
from rtvoice.tools.views import SpecialToolParameters


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

    def __init__(self):
        self._special_param_names = set(SpecialToolParameters.model_fields.keys())
        self._special_param_types = self._build_special_param_types()

    def build(self, func: Callable) -> FunctionParameters:
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

    def _should_skip_param(self, param_name: str, type_hints: dict[str, Any]) -> bool:
        if param_name in ("self", "cls", *self._special_param_names):
            return True

        param_type = type_hints.get(param_name)
        return param_type and self._is_special_type(param_type)

    def _build_special_param_types(self) -> set[type]:
        special_types: set[type] = set()

        for field in SpecialToolParameters.model_fields.values():
            resolved_type = self._unwrap_optional(field.annotation)
            if isinstance(resolved_type, type):
                special_types.add(resolved_type)

        return special_types

    def _is_special_type(self, type_hint: Any) -> bool:
        actual_type, _ = self._extract_type_and_description(type_hint)
        origin = get_origin(actual_type)

        if origin is Union:
            non_none_args = [
                arg for arg in get_args(actual_type) if arg is not type(None)
            ]
            return any(self._is_special_type(arg) for arg in non_none_args)

        return (
            isinstance(actual_type, type) and actual_type in self._special_param_types
        )

    def _extract_type_and_description(self, type_hint: Any) -> tuple[Any, str | None]:
        if get_origin(type_hint) is not Annotated:
            return type_hint, None

        args = get_args(type_hint)
        actual_type = args[0]
        description = next((arg for arg in args[1:] if isinstance(arg, str)), None)

        return actual_type, description

    def _unwrap_optional(self, type_hint: Any) -> Any:
        origin = get_origin(type_hint)
        if origin is Union:
            non_none_types = [
                arg for arg in get_args(type_hint) if arg is not type(None)
            ]
            if non_none_types:
                return non_none_types[0]
        return type_hint

    def _convert_to_json_schema(
        self, python_type: Any, description: str | None = None
    ) -> FunctionParameterProperty:
        origin = get_origin(python_type)

        if origin is Union:
            return self._handle_union_type(python_type, description)

        if origin is list:
            return FunctionParameterProperty(type="array", description=description)

        if origin is dict:
            return FunctionParameterProperty(type="object", description=description)

        if origin in self._COLLECTION_TYPES:
            return FunctionParameterProperty(type="array", description=description)

        json_type = self._PRIMITIVE_TYPES.get(python_type)
        if json_type:
            return FunctionParameterProperty(type=json_type, description=description)

        if self._is_pydantic_model(python_type):
            return FunctionParameterProperty(type="object", description=description)

        return FunctionParameterProperty(type="string", description=description)

    def _handle_union_type(
        self, union_type: Any, description: str | None
    ) -> FunctionParameterProperty:
        non_none_args = [arg for arg in get_args(union_type) if arg is not type(None)]

        if len(non_none_args) == 1:
            return self._convert_to_json_schema(non_none_args[0], description)

        return FunctionParameterProperty(type="string", description=description)

    def _is_pydantic_model(self, python_type: Any) -> bool:
        return isinstance(python_type, type) and issubclass(python_type, BaseModel)
