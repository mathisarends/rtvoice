import json
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from rtvoice.llm import (
    AssistantMessage,
    ChatInvokeCompletion,
    ContentPartImageParam,
    ContentPartTextParam,
    Function,
    ImageURL,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from rtvoice.llm.providers.openai_compatible import BaseOpenAICompatible
from rtvoice.llm.tools import FunctionTool


class SearchResult(BaseModel):
    query: str
    results: list[str]


class MockChatModel(BaseOpenAICompatible):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = AsyncMock()
        self._model = "gpt-4"


@pytest.fixture
def mock_model() -> MockChatModel:
    return MockChatModel()


@pytest.fixture
def search_tool() -> FunctionTool:
    def search_web(query: str) -> str:
        """Search the web"""
        return f"Results for: {query}"

    return FunctionTool(search_web)


class TestMessageConversion:
    def test_converts_user_message(self, mock_model: MockChatModel) -> None:
        messages = [UserMessage(content="Hello")]
        converted = mock_model._convert_messages(messages)

        assert converted == [{"role": "user", "content": "Hello"}]

    def test_converts_system_message(self, mock_model: MockChatModel) -> None:
        messages = [SystemMessage(content="You are helpful")]
        converted = mock_model._convert_messages(messages)

        assert converted == [{"role": "system", "content": "You are helpful"}]

    def test_converts_assistant_message(self, mock_model: MockChatModel) -> None:
        messages = [AssistantMessage(content="I can help")]
        converted = mock_model._convert_messages(messages)

        assert converted == [{"role": "assistant", "content": "I can help"}]

    def test_converts_user_message_with_image(self, mock_model: MockChatModel) -> None:
        message = UserMessage(
            content=[
                ContentPartTextParam(text="What's this?"),
                ContentPartImageParam(
                    image_url=ImageURL(
                        url="data:image/png;base64,iVBORw0KG...", media_type="image/png"
                    )
                ),
            ]
        )
        converted = mock_model._convert_single_message(message)

        assert converted["role"] == "user"
        assert len(converted["content"]) == 2
        assert converted["content"][0]["type"] == "text"
        assert converted["content"][1]["type"] == "image_url"
        assert "data:image/png;base64," in converted["content"][1]["image_url"]["url"]

    def test_converts_tool_result_message(self, mock_model: MockChatModel) -> None:
        message = ToolResultMessage(tool_call_id="call_123", content="Search completed")
        converted = mock_model._convert_single_message(message)

        assert converted == {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "Search completed",
        }

    def test_converts_assistant_message_with_tool_calls(
        self, mock_model: MockChatModel
    ) -> None:
        message = AssistantMessage(
            content="Let me search",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    function=Function(
                        name="search", arguments=json.dumps({"query": "test"})
                    ),
                )
            ],
        )
        converted = mock_model._convert_single_message(message)

        assert converted["role"] == "assistant"
        assert converted["content"] == "Let me search"
        assert len(converted["tool_calls"]) == 1
        assert converted["tool_calls"][0]["id"] == "call_123"
        assert converted["tool_calls"][0]["function"]["name"] == "search"

    def test_converts_assistant_message_with_serialized_pydantic_arguments(
        self, mock_model: MockChatModel
    ) -> None:
        class SearchParams(BaseModel):
            query: str
            max_results: int

        params = SearchParams(query="test", max_results=10)
        message = AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_456",
                    function=Function(
                        name="search", arguments=params.model_dump_json()
                    ),
                )
            ],
        )
        converted = mock_model._convert_single_message(message)

        parsed_args = json.loads(converted["tool_calls"][0]["function"]["arguments"])
        assert converted["tool_calls"][0]["function"]["name"] == "search"
        assert parsed_args["query"] == "test"
        assert parsed_args["max_results"] == 10


class TestParameterMerging:
    def test_merges_default_and_method_params(self) -> None:
        model = MockChatModel(temperature=0.5, max_tokens=100)
        params = model._merge_params({"top_p": 0.9})

        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 100
        assert params["top_p"] == 0.9

    def test_method_params_override_defaults(self) -> None:
        model = MockChatModel(temperature=0.5)
        params = model._merge_params({"temperature": 0.8})

        assert params["temperature"] == 0.8

    def test_excludes_none_values(self) -> None:
        model = MockChatModel(temperature=0.5)
        params = model._merge_params({"top_p": 0.9})

        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    def test_preserves_custom_kwargs(self) -> None:
        model = MockChatModel(custom_param="value")
        params = model._merge_params({})

        assert params["custom_param"] == "value"


class TestPlainInvoke:
    @pytest.mark.asyncio
    async def test_returns_completion_with_text_content(
        self, mock_model: MockChatModel
    ) -> None:
        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [
            Mock(message=Mock(content="Hello world"), finish_reason="stop")
        ]
        mock_response.usage = None
        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await mock_model.invoke([UserMessage(content="Hi")])

        assert result.completion == "Hello world"

    @pytest.mark.asyncio
    async def test_returns_empty_string_for_none_content(
        self, mock_model: MockChatModel
    ) -> None:
        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [Mock(message=Mock(content=None), finish_reason="stop")]
        mock_response.usage = None
        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await mock_model.invoke([UserMessage(content="Hi")])

        assert result.completion == ""

    @pytest.mark.asyncio
    async def test_passes_merged_parameters(self, mock_model: MockChatModel) -> None:
        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [
            Mock(message=Mock(content="Response"), finish_reason="stop")
        ]
        mock_response.usage = None
        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        await mock_model.invoke(
            [UserMessage(content="Hi")], temperature=0.7, max_tokens=50
        )

        call_kwargs = mock_model._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 50


class TestToolInvoke:
    @pytest.mark.asyncio
    async def test_returns_completion_with_tool_calls(
        self, mock_model: MockChatModel, search_tool: FunctionTool
    ) -> None:
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "search_web"
        mock_tool_call.function.arguments = '{"query": "test"}'

        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [
            Mock(
                message=Mock(content="Searching...", tool_calls=[mock_tool_call]),
                finish_reason="tool_calls",
            )
        ]
        mock_response.usage = None
        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await mock_model.invoke(
            [UserMessage(content="Search for test")], tools=[search_tool]
        )

        assert isinstance(result, ChatInvokeCompletion)
        assert result.completion == "Searching..."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search_web"
        assert json.loads(result.tool_calls[0].function.arguments) == {"query": "test"}

    @pytest.mark.asyncio
    async def test_returns_empty_tool_calls_when_none(
        self, mock_model: MockChatModel, search_tool: FunctionTool
    ) -> None:
        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [
            Mock(message=Mock(content="Done", tool_calls=None), finish_reason="stop")
        ]
        mock_response.usage = None
        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await mock_model.invoke(
            [UserMessage(content="Hi")], tools=[search_tool]
        )

        assert isinstance(result, ChatInvokeCompletion)
        assert result.completion == "Done"
        assert result.tool_calls == []


class TestStructuredOutput:
    @pytest.mark.asyncio
    async def test_returns_parsed_pydantic_model(
        self, mock_model: MockChatModel
    ) -> None:
        expected_result = SearchResult(query="test", results=["a", "b"])
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(parsed=expected_result), finish_reason="stop")
        ]
        mock_response.usage = None
        mock_model._client.beta.chat.completions.parse = AsyncMock(
            return_value=mock_response
        )

        result = await mock_model.invoke(
            [UserMessage(content="Search")], output_format=SearchResult
        )

        assert isinstance(result.completion, SearchResult)
        assert result.completion.query == "test"
        assert result.completion.results == ["a", "b"]

    @pytest.mark.asyncio
    async def test_uses_beta_parse_endpoint_with_response_format(
        self, mock_model: MockChatModel
    ) -> None:
        expected_result = SearchResult(query="test", results=[])
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(parsed=expected_result), finish_reason="stop")
        ]
        mock_response.usage = None
        mock_model._client.beta.chat.completions.parse = AsyncMock(
            return_value=mock_response
        )

        await mock_model.invoke(
            [UserMessage(content="Search")], output_format=SearchResult
        )

        mock_model._client.beta.chat.completions.parse.assert_called_once()
        call_kwargs = mock_model._client.beta.chat.completions.parse.call_args.kwargs
        assert call_kwargs["response_format"] is SearchResult


class TestStreaming:
    @pytest.mark.asyncio
    async def test_yields_content_chunks(self, mock_model: MockChatModel) -> None:
        mock_chunk1 = Mock(spec=ChatCompletionChunk)
        mock_chunk1.choices = [Mock(delta=Mock(content="Hello"))]
        mock_chunk2 = Mock(spec=ChatCompletionChunk)
        mock_chunk2.choices = [Mock(delta=Mock(content=" world"))]
        mock_chunk3 = Mock(spec=ChatCompletionChunk)
        mock_chunk3.choices = [Mock(delta=Mock(content=None))]

        async def mock_stream():
            for chunk in [mock_chunk1, mock_chunk2, mock_chunk3]:
                yield chunk

        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        chunks = []
        async for chunk in mock_model.stream([UserMessage(content="Hi")]):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_passes_stream_parameter(self, mock_model: MockChatModel) -> None:
        async def mock_stream():
            if False:
                yield

        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        async for _ in mock_model.stream([UserMessage(content="Hi")]):
            pass

        call_kwargs = mock_model._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
