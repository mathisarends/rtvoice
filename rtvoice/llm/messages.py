from dataclasses import dataclass
from enum import StrEnum


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    role: MessageRole
    content: str
    tool_call_id: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    result: str | None = None


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_message(self) -> ChatMessage:
        return ChatMessage(role="assistant", content=self.content)
