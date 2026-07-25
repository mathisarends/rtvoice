from pydantic import BaseModel
from transitbus import EventBus

from rtvoice.tools.di import Inject
from rtvoice.tools.schemas import build


class SampleModel(BaseModel):
    name: str
    value: int


class TestPrimitiveTypes:
    def test_string_param(self) -> None:
        def func(name: str) -> None: ...

        result = build(func)

        assert result.properties["name"].type == "string"

    def test_int_param(self) -> None:
        def func(count: int) -> None: ...

        result = build(func)

        assert result.properties["count"].type == "integer"

    def test_float_param(self) -> None:
        def func(score: float) -> None: ...

        result = build(func)

        assert result.properties["score"].type == "number"

    def test_bool_param(self) -> None:
        def func(enabled: bool) -> None: ...

        result = build(func)

        assert result.properties["enabled"].type == "boolean"

    def test_list_param(self) -> None:
        def func(items: list) -> None: ...

        result = build(func)

        assert result.properties["items"].type == "array"

    def test_dict_param(self) -> None:
        def func(data: dict) -> None: ...

        result = build(func)

        assert result.properties["data"].type == "object"


class TestOptionalTypes:
    def test_optional_string_resolves_to_string(self) -> None:
        def func(name: str | None = None) -> None: ...

        result = build(func)

        assert result.properties["name"].type == "string"

    def test_optional_int_resolves_to_integer(self) -> None:
        def func(count: int | None = None) -> None: ...

        result = build(func)

        assert result.properties["count"].type == "integer"

    def test_union_with_multiple_types_falls_back_to_string(self) -> None:
        def func(value: int | str) -> None: ...

        result = build(func)

        assert result.properties["value"].type == "string"


class TestRequiredParams:
    def test_param_without_default_is_required(self) -> None:
        def func(name: str) -> None: ...

        result = build(func)

        assert "name" in result.required

    def test_param_with_default_is_not_required(self) -> None:
        def func(name: str = "default") -> None: ...

        result = build(func)

        assert "name" not in result.required

    def test_optional_param_is_not_required(self) -> None:
        def func(name: str | None = None) -> None: ...

        result = build(func)

        assert "name" not in result.required

    def test_mixed_required_and_optional(self) -> None:
        def func(required: str, optional: str = "default") -> None: ...

        result = build(func)

        assert "required" in result.required
        assert "optional" not in result.required


class TestSkippedParams:
    def test_self_param_is_skipped(self) -> None:
        class MyClass:
            def method(self, name: str) -> None: ...

        result = build(MyClass.method)

        assert "self" not in result.properties

    def test_cls_param_is_skipped(self) -> None:
        class MyClass:
            @classmethod
            def method(cls, name: str) -> None: ...

        result = build(MyClass.method)

        assert "cls" not in result.properties

    def test_inject_param_is_skipped(self) -> None:
        def func(name: str, event_bus: Inject[EventBus]) -> None: ...

        result = build(func)

        assert "event_bus" not in result.properties
        assert "name" in result.properties


class TestPydanticModelParams:
    def test_pydantic_model_maps_to_object(self) -> None:
        def func(data: SampleModel) -> None: ...

        result = build(func)

        assert result.properties["data"].type == "object"


class TestCollectionTypes:
    def test_typed_list_maps_to_array(self) -> None:
        def func(items: list[str]) -> None: ...

        result = build(func)

        assert result.properties["items"].type == "array"

    def test_typed_dict_maps_to_object(self) -> None:
        def func(mapping: dict[str, int]) -> None: ...

        result = build(func)

        assert result.properties["mapping"].type == "object"


class TestSchemaMetadata:
    def test_schema_type_is_object(self) -> None:
        def func(name: str) -> None: ...

        result = build(func)

        assert result.type == "object"

    def test_schema_strict_is_true(self) -> None:
        def func(name: str) -> None: ...

        result = build(func)

        assert result.strict is True

    def test_empty_function_produces_empty_schema(self) -> None:
        def func() -> None: ...

        result = build(func)

        assert result.properties == {}
        assert result.required == []
