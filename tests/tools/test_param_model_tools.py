from enum import StrEnum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field

from rtvoice.events.bus import EventBus
from rtvoice.tools import Tools
from rtvoice.tools.di import Inject, ToolContext
from rtvoice.tools.views import ToolSchemaBuilder


class SearchParams(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=10, description="Maximum results to return")


class CreateEventParams(BaseModel):
    title: str = Field(description="Event title")
    date: str = Field(description="Event date")
    attendees: list[str] = Field(default_factory=list, description="List of attendees")


class Priority(StrEnum):
    LOW = "low"
    HIGH = "high"


class LocationParams(BaseModel):
    city: str = Field(description="City name")
    country: str


class ReminderParams(BaseModel):
    title: Annotated[str, "Reminder title"]
    priority: Priority
    channel: Literal["email", "sms"] = "email"
    locations: list[LocationParams] = Field(default_factory=list)


@pytest.fixture
def tools() -> Tools:
    return Tools()


@pytest.fixture
def builder() -> ToolSchemaBuilder:
    return ToolSchemaBuilder()


class TestSchemaBuilderFromModel:
    def test_builds_properties_from_model_fields(
        self, builder: ToolSchemaBuilder
    ) -> None:
        schema = builder.build_from_model(SearchParams)

        assert "query" in schema.properties
        assert "max_results" in schema.properties

    def test_maps_field_types_correctly(self, builder: ToolSchemaBuilder) -> None:
        schema = builder.build_from_model(SearchParams)

        assert schema.properties["query"].type == "string"
        assert schema.properties["max_results"].type == "integer"

    def test_extracts_field_descriptions(self, builder: ToolSchemaBuilder) -> None:
        schema = builder.build_from_model(SearchParams)

        assert schema.properties["query"].description == "The search query"

    def test_required_fields_are_marked(self, builder: ToolSchemaBuilder) -> None:
        schema = builder.build_from_model(SearchParams)

        assert "query" in schema.required
        assert "max_results" not in schema.required

    def test_default_values_are_included(self, builder: ToolSchemaBuilder) -> None:
        schema = builder.build_from_model(SearchParams)

        assert schema.properties["max_results"].default == 10

    def test_default_factory_is_not_included_as_default(
        self, builder: ToolSchemaBuilder
    ) -> None:
        schema = builder.build_from_model(CreateEventParams)

        assert schema.properties["attendees"].default is None

    def test_list_field_maps_to_array(self, builder: ToolSchemaBuilder) -> None:
        schema = builder.build_from_model(CreateEventParams)

        assert schema.properties["attendees"].type == "array"

    def test_typed_list_field_includes_item_schema(
        self, builder: ToolSchemaBuilder
    ) -> None:
        schema = builder.build_from_model(CreateEventParams)

        assert schema.properties["attendees"].items is not None
        assert schema.properties["attendees"].items.type == "string"

    def test_nested_model_field_includes_object_schema(
        self, builder: ToolSchemaBuilder
    ) -> None:
        schema = builder.build_from_model(ReminderParams)
        locations = schema.properties["locations"]

        assert locations.items is not None
        assert locations.items.type == "object"
        assert locations.items.properties is not None
        assert locations.items.properties["city"].description == "City name"
        assert locations.items.required == ["city", "country"]

    def test_enum_and_literal_fields_include_allowed_values(
        self, builder: ToolSchemaBuilder
    ) -> None:
        schema = builder.build_from_model(ReminderParams)

        assert schema.properties["priority"].enum == ["low", "high"]
        assert schema.properties["channel"].enum == ["email", "sms"]

    def test_annotated_field_uses_description_metadata(
        self, builder: ToolSchemaBuilder
    ) -> None:
        schema = builder.build_from_model(ReminderParams)

        assert schema.properties["title"].description == "Reminder title"

    def test_build_delegates_to_build_from_model(
        self, builder: ToolSchemaBuilder
    ) -> None:
        def func(params: SearchParams) -> None: ...

        schema = builder.build(func, param_model=SearchParams)

        assert "query" in schema.properties
        assert "max_results" in schema.properties
        assert "params" not in schema.properties


class TestActionWithParamModel:
    def test_registers_tool_with_param_model(self, tools: Tools) -> None:
        @tools.action(description="Search", param_model=SearchParams)
        async def search(params: SearchParams) -> str:
            return params.query

        tool = tools.get("search")
        assert tool is not None
        assert tool.param_model is SearchParams

    def test_schema_is_built_from_param_model(self, tools: Tools) -> None:
        @tools.action(description="Search", param_model=SearchParams)
        async def search(params: SearchParams) -> str:
            return params.query

        tool = tools.get("search")
        assert "query" in tool.schema.properties
        assert "params" not in tool.schema.properties

    @pytest.mark.asyncio
    async def test_flat_params_still_work(self, tools: Tools) -> None:
        @tools.action(description="Add numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        result = await tools.execute("add", {"a": 3, "b": 7})

        assert result == 10

    @pytest.mark.asyncio
    async def test_executes_with_param_model(self, tools: Tools) -> None:
        @tools.action(description="Search", param_model=SearchParams)
        async def search(params: SearchParams) -> str:
            return f"{params.query}:{params.max_results}"

        result = await tools.execute("search", {"query": "test", "max_results": 5})

        assert result == "test:5"

    @pytest.mark.asyncio
    async def test_param_model_with_defaults(self, tools: Tools) -> None:
        @tools.action(description="Search", param_model=SearchParams)
        async def search(params: SearchParams) -> int:
            return params.max_results

        result = await tools.execute("search", {"query": "test"})

        assert result == 10

    @pytest.mark.asyncio
    async def test_param_model_with_inject(self, tools: Tools) -> None:
        injected_bus = EventBus()
        tools.set_context(ToolContext(event_bus=injected_bus))

        received = {}

        @tools.action(description="Search", param_model=SearchParams)
        async def search(params: SearchParams, bus: Inject[EventBus]) -> None:
            received["params"] = params
            received["bus"] = bus

        await tools.execute("search", {"query": "hello"})

        assert isinstance(received["params"], SearchParams)
        assert received["params"].query == "hello"
        assert received["bus"] is injected_bus


class TestLambdaStatus:
    def test_lambda_status_with_param_model(self, tools: Tools) -> None:
        @tools.action(
            description="Search",
            param_model=SearchParams,
            status=lambda p: f"Searching for '{p.query}'",
        )
        async def search(params: SearchParams) -> str:
            return params.query

        tool = tools.get("search")
        status = tool.format_status({"query": "dentist", "max_results": 5})

        assert status == "Searching for 'dentist'"

    def test_lambda_status_with_pydantic_instance(self, tools: Tools) -> None:
        @tools.action(
            description="Search",
            param_model=SearchParams,
            status=lambda p: f"Searching for '{p.query}'",
        )
        async def search(params: SearchParams) -> str:
            return params.query

        tool = tools.get("search")
        status = tool.format_status(SearchParams(query="dentist"))

        assert status == "Searching for 'dentist'"

    def test_lambda_status_without_param_model_raises(self, tools: Tools) -> None:
        with pytest.raises(ValueError, match="status requires a param_model"):

            @tools.action(
                description="Search",
                status=lambda p: f"Searching '{p.query}'",
            )
            async def search(query: str) -> str:
                return query

    def test_lambda_status_accessing_nonexistent_field_raises(
        self, tools: Tools
    ) -> None:
        with pytest.raises(ValueError, match="does not exist"):

            @tools.action(
                description="Search",
                param_model=SearchParams,
                status=lambda p: f"Looking for '{p.nonexistent}'",
            )
            async def search(params: SearchParams) -> str:
                return params.query

    def test_lambda_status_returning_non_string_raises(self, tools: Tools) -> None:
        with pytest.raises(ValueError, match="must return str"):

            @tools.action(
                description="Search",
                param_model=SearchParams,
                status=lambda p: 42,
            )
            async def search(params: SearchParams) -> str:
                return params.query

    def test_string_status_without_param_model_raises(self, tools: Tools) -> None:
        with pytest.raises(ValueError, match="status requires a param_model"):

            @tools.action(
                description="Search",
                status="Searching for '{query}'",
            )
            async def search(query: str) -> str:
                return query

    def test_string_status_with_param_model(self, tools: Tools) -> None:
        @tools.action(
            description="Search",
            param_model=SearchParams,
            status="Searching for '{query}'",
        )
        async def search(params: SearchParams) -> str:
            return params.query

        tool = tools.get("search")
        status = tool.format_status({"query": "dentist"})

        assert status == "Searching for 'dentist'"

    def test_string_status_with_unknown_placeholder_raises(self, tools: Tools) -> None:
        with pytest.raises(ValueError, match="unknown placeholders"):

            @tools.action(
                description="Search",
                param_model=SearchParams,
                status="Searching for '{nonexistent}'",
            )
            async def search(params: SearchParams) -> str:
                return params.query

    def test_string_status_validates_against_param_model_fields(
        self, tools: Tools
    ) -> None:
        with pytest.raises(ValueError, match="unknown placeholders"):

            @tools.action(
                description="Search",
                param_model=SearchParams,
                status="Looking for '{nonexistent}'",
            )
            async def search(params: SearchParams) -> str:
                return params.query

    def test_no_status_returns_none(self, tools: Tools) -> None:
        @tools.action(description="Search")
        async def search(query: str) -> str:
            return query

        tool = tools.get("search")
        status = tool.format_status({"query": "dentist"})

        assert status is None

    def test_lambda_status_with_create_event_params(self, tools: Tools) -> None:
        @tools.action(
            description="Create event",
            param_model=CreateEventParams,
            status=lambda p: f"Creating '{p.title}' on {p.date}",
        )
        async def create_event(params: CreateEventParams) -> str:
            return params.title

        tool = tools.get("create_event")
        status = tool.format_status({"title": "Meeting", "date": "2026-04-10"})

        assert status == "Creating 'Meeting' on 2026-04-10"
