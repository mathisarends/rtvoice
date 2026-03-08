from typing import Protocol

from rtvoice.llm.messages import ChatMessage, LLMResponse


class ChatModel(Protocol):
    async def invoke(
        self, messages: list[ChatMessage], tools: list[dict] | None = None
    ) -> LLMResponse: ...
