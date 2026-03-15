from rtvoice.llm.tools import FunctionTool, tool


def sample_function(query: str, max_results: int = 10) -> str:
    """Search for information"""
    return f"Searching: {query}"


def no_hints_function(query, count=5):
    return f"{query}: {count}"


class TestFunctionTool:
    def test_extracts_name_from_function(self) -> None:
        t = FunctionTool(sample_function)
        assert t.name == "sample_function"

    def test_uses_custom_name_when_provided(self) -> None:
        t = FunctionTool(sample_function, name="custom_search")
        assert t.name == "custom_search"

    def test_extracts_description_from_docstring(self) -> None:
        t = FunctionTool(sample_function)
        schema = t.to_openai_schema()
        assert schema["function"]["description"] == "Search for information"

    def test_generates_valid_openai_schema(self) -> None:
        t = FunctionTool(sample_function)
        schema = t.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "sample_function"
        assert "parameters" in schema["function"]

    def test_identifies_required_parameters(self) -> None:
        t = FunctionTool(sample_function)
        params = t.to_openai_schema()["function"]["parameters"]

        assert params["required"] == ["query"]
        assert "max_results" not in params["required"]

    def test_maps_python_types_to_json_schema(self) -> None:
        t = FunctionTool(sample_function)
        props = t.to_openai_schema()["function"]["parameters"]["properties"]

        assert props["query"]["type"] == "string"
        assert props["max_results"]["type"] == "integer"

    def test_defaults_to_string_when_no_type_hints(self) -> None:
        t = FunctionTool(no_hints_function)
        props = t.to_openai_schema()["function"]["parameters"]["properties"]

        assert props["query"]["type"] == "string"
        assert props["count"]["type"] == "string"

    def test_parses_json_arguments(self) -> None:
        t = FunctionTool(sample_function)
        args = t.parse_arguments('{"query": "test", "max_results": 5}')

        assert args == {"query": "test", "max_results": 5}

    def test_remains_callable(self) -> None:
        t = FunctionTool(sample_function)
        result = t("test query", max_results=3)

        assert result == "Searching: test query"

    def test_ignores_self_and_cls_parameters(self) -> None:
        class MyClass:
            def method(self, query: str) -> str:
                return query

        t = FunctionTool(MyClass().method)
        props = t.to_openai_schema()["function"]["parameters"]["properties"]

        assert "self" not in props
        assert "query" in props


class TestToolDecorator:
    def test_converts_function_to_tool(self) -> None:
        @tool
        def search(query: str) -> str:
            """Search function"""
            return query

        assert isinstance(search, FunctionTool)
        assert search.name == "search"

    def test_accepts_custom_name(self) -> None:
        @tool(name="web_search")
        def search(query: str) -> str:
            return query

        assert search.name == "web_search"

    def test_accepts_custom_description(self) -> None:
        @tool(description="Custom desc")
        def search(query: str) -> str:
            """Original doc"""
            return query

        schema = search.to_openai_schema()
        assert schema["function"]["description"] == "Custom desc"

    def test_preserves_function_behavior(self) -> None:
        @tool
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5
