from collections.abc import AsyncIterator
from typing import Any, Literal, TypeVar, overload

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from pydantic import BaseModel

from rtvoice.llm.base import BaseChatModel
from rtvoice.llm.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    Function,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from rtvoice.llm.tools import Tool
from rtvoice.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar("T", bound=BaseModel)


class BaseOpenAICompatible(BaseChatModel):
    _client: AsyncOpenAI | AsyncAzureOpenAI
    _model: str

    @overload
    async def invoke[T](
        self, messages: list[Message], output_format: type[T], **kwargs: Any
    ) -> ChatInvokeCompletion[T]: ...

    @overload
    async def invoke(
        self, messages: list[Message], output_format: None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[str]: ...

    async def invoke[T](
        self,
        messages: list[Message],
        output_format: type[T] | None = None,
        tools: list[Tool | dict] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        params = self._merge_params(kwargs)
        converted_messages = self._convert_messages(messages)

        if output_format is not None:
            return await self._invoke_with_structured_output(
                converted_messages, output_format, params
            )

        if tools:
            return await self._invoke_with_tools(
                converted_messages, tools, params, tool_choice
            )

        return await self._invoke_plain(converted_messages, params)

    async def _invoke_with_structured_output[T](
        self, messages: list[dict], output_format: type[T], params: dict[str, Any]
    ) -> ChatInvokeCompletion[T]:
        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=output_format,
            **params,
        )
        choice = response.choices[0]
        usage = response.usage
        return ChatInvokeCompletion(
            completion=choice.message.parsed,
            stop_reason=choice.finish_reason,
            usage=self._parse_usage(usage),
        )

    async def _invoke_with_tools(
        self,
        messages: list[dict],
        tools: list[Tool | dict],
        params: dict[str, Any],
        tool_choice: Literal["auto", "required", "none"] = "auto",
    ) -> ChatInvokeCompletion[str]:
        openai_tools = [
            t if isinstance(t, dict) else t.to_openai_schema() for t in tools
        ]
        response: ChatCompletion = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=openai_tools,
            tool_choice=tool_choice,
            **params,
        )
        choice = response.choices[0]
        return ChatInvokeCompletion(
            completion=choice.message.content or "",
            tool_calls=self._parse_tool_calls(choice.message.tool_calls),
            stop_reason=choice.finish_reason,
            usage=self._parse_usage(response.usage),
        )

    async def _invoke_plain(
        self, messages: list[dict], params: dict[str, Any]
    ) -> ChatInvokeCompletion[str]:
        response: ChatCompletion = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **params,
        )
        choice = response.choices[0]
        return ChatInvokeCompletion(
            completion=choice.message.content or "",
            stop_reason=choice.finish_reason,
            usage=self._parse_usage(response.usage),
        )

    def _parse_usage(self, usage: CompletionUsage | None) -> ChatInvokeUsage | None:
        if not usage:
            return None
        return ChatInvokeUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            prompt_cached_tokens=getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", None
            ),
        )

    def _parse_tool_calls(
        self, raw_tool_calls: list[ChatCompletionMessageToolCall] | None
    ) -> list[ToolCall]:
        if not raw_tool_calls:
            return []
        return [
            ToolCall(
                id=tc.id,
                function=Function(
                    name=tc.function.name, arguments=tc.function.arguments
                ),
            )
            for tc in raw_tool_calls
        ]

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        return [self._convert_single_message(msg) for msg in messages]

    def _convert_single_message(self, msg: Message) -> dict:
        if isinstance(msg, ToolResultMessage):
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            }

        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            return {
                "role": "assistant",
                "content": msg.text or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }

        if isinstance(msg, UserMessage) and isinstance(msg.content, list):
            content = []
            for part in msg.content:
                if isinstance(part, ContentPartTextParam):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, ContentPartImageParam):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": part.image_url.url,
                                "detail": part.image_url.detail,
                            },
                        }
                    )
            return {"role": msg.role, "content": content}

        if isinstance(msg, (UserMessage, SystemMessage)):
            if isinstance(msg.content, list):
                return {
                    "role": msg.role,
                    "content": [{"type": "text", "text": p.text} for p in msg.content],
                }
            return {"role": msg.role, "content": msg.content}

        return {"role": msg.role, "content": msg.text}

    async def stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        params = self._merge_params(kwargs)
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=self._convert_messages(messages),
            stream=True,
            **params,
        )
        chunk: ChatCompletionChunk
        async for chunk in stream:
            if content := chunk.choices[0].delta.content:
                yield content
