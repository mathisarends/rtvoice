import json

import pytest

from rtvoice.llm.tools import RawSchemaTool


class TestRawSchemaTool:
    def test_stores_name(self) -> None:
        t = RawSchemaTool(
            name="search_web", schema={"type": "object", "properties": {}}
        )
        assert t.name == "search_web"

    def test_stores_description(self) -> None:
        t = RawSchemaTool(name="search", schema={}, description="Search the web")
        schema = t.to_openai_schema()
        assert schema["function"]["description"] == "Search the web"

    def test_defaults_to_empty_description(self) -> None:
        t = RawSchemaTool(name="search", schema={})
        schema = t.to_openai_schema()
        assert schema["function"]["description"] == ""

    def test_generates_valid_openai_schema(self) -> None:
        t = RawSchemaTool(
            name="calculator",
            schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        )
        schema = t.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculator"
        assert schema["function"]["parameters"]["type"] == "object"
        assert "a" in schema["function"]["parameters"]["properties"]

    def test_preserves_schema_structure(self) -> None:
        original_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        }

        t = RawSchemaTool(name="search", schema=original_schema)
        result = t.to_openai_schema()

        assert result["function"]["parameters"] == original_schema

    def test_parses_json_arguments(self) -> None:
        t = RawSchemaTool(name="search", schema={})
        args = t.parse_arguments('{"query": "test", "max": 10}')

        assert args == {"query": "test", "max": 10}

    def test_raises_on_invalid_json(self) -> None:
        t = RawSchemaTool(name="search", schema={})

        with pytest.raises(json.JSONDecodeError):
            t.parse_arguments("invalid json")

    def test_handles_complex_nested_schema(self) -> None:
        t = RawSchemaTool(
            name="complex_tool",
            schema={
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                    }
                },
            },
        )

        schema = t.to_openai_schema()
        filters = schema["function"]["parameters"]["properties"]["filters"]

        assert filters["type"] == "object"
        assert "category" in filters["properties"]
