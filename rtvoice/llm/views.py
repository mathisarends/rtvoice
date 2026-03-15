from llmify.messages import ToolCall
from pydantic import BaseModel


class ChatInvokeUsage(BaseModel):
    prompt_tokens: int
    prompt_cached_tokens: int | None = None
    completion_tokens: int
    total_tokens: int


class ChatInvokeCompletion[T](BaseModel):
    completion: T
    thinking: str | None = None
    redacted_thinking: str | None = None
    usage: ChatInvokeUsage | None = None
    stop_reason: str | None = None
    tool_calls: list[ToolCall] = []
