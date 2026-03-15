from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


def _truncate(text: str, max_length: int = 50) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class ContentPartTextParam(BaseModel):
    text: str
    type: Literal["text"] = "text"

    def __str__(self) -> str:
        return f"Text: {_truncate(self.text)}"

    def __repr__(self) -> str:
        return f"ContentPartTextParam(text={_truncate(self.text)})"


class ContentPartRefusalParam(BaseModel):
    refusal: str
    type: Literal["refusal"] = "refusal"

    def __str__(self) -> str:
        return f"Refusal: {_truncate(self.refusal)}"

    def __repr__(self) -> str:
        return f"ContentPartRefusalParam(refusal={_truncate(repr(self.refusal), 50)})"


SupportedImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class ImageURL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"
    # needed for Anthropic
    media_type: SupportedImageMediaType = "image/png"

    @staticmethod
    def _format_url(url: str, max_length: int = 50) -> str:
        if url.startswith("data:"):
            media_type = url.split(";")[0].split(":")[1] if ";" in url else "image"
            return f"<base64 {media_type}>"
        return _truncate(url, max_length)

    def __str__(self) -> str:
        url_display = self._format_url(self.url)
        return f"🖼️  Image[{self.media_type}, detail={self.detail}]: {url_display}"

    def __repr__(self) -> str:
        url_repr = self._format_url(self.url, 30)
        return f"ImageURL(url={url_repr!r}, detail={self.detail!r}, media_type={self.media_type!r})"


class ContentPartImageParam(BaseModel):
    image_url: ImageURL
    type: Literal["image_url"] = "image_url"

    def __str__(self) -> str:
        return str(self.image_url)

    def __repr__(self) -> str:
        return f"ContentPartImageParam(image_url={self.image_url!r})"


class Function(BaseModel):
    arguments: str
    name: str

    def __str__(self) -> str:
        args_preview = _truncate(self.arguments, 80)
        return f"{self.name}({args_preview})"

    def __repr__(self) -> str:
        args_repr = _truncate(repr(self.arguments), 50)
        return f"Function(name={self.name!r}, arguments={args_repr})"


class ToolCall(BaseModel):
    id: str
    function: Function
    type: Literal["function"] = "function"

    def __str__(self) -> str:
        return f"ToolCall[{self.id}]: {self.function}"

    def __repr__(self) -> str:
        return f"ToolCall(id={self.id!r}, function={self.function!r})"


class _MessageRole(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"


class _MessageBase(BaseModel):
    role: _MessageRole

    cache: bool = False
    """Whether to cache this message. This is only applicable when using Anthropic models."""


class UserMessage(_MessageBase):
    role: _MessageRole = _MessageRole.USER
    content: str | list[ContentPartTextParam | ContentPartImageParam]
    name: str | None = None

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            return "\n".join(
                [part.text for part in self.content if part.type == "text"]
            )
        else:
            return ""

    def __str__(self) -> str:
        return f"UserMessage(content={self.text})"

    def __repr__(self) -> str:
        return f"UserMessage(content={self.text!r})"


class SystemMessage(_MessageBase):
    role: _MessageRole = _MessageRole.SYSTEM
    content: str | list[ContentPartTextParam]
    name: str | None = None

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            return "\n".join(
                [part.text for part in self.content if part.type == "text"]
            )
        else:
            return ""

    def __str__(self) -> str:
        return f"SystemMessage(content={self.text})"

    def __repr__(self) -> str:
        return f"SystemMessage(content={self.text!r})"


class AssistantMessage(_MessageBase):
    role: _MessageRole = _MessageRole.ASSISTANT
    content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None = None
    name: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] = []

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            text = ""
            for part in self.content:
                if part.type == "text":
                    text += part.text
                elif part.type == "refusal":
                    text += f"[Refusal] {part.refusal}"
            return text
        else:
            return ""

    def __str__(self) -> str:
        return f"AssistantMessage(content={self.text})"

    def __repr__(self) -> str:
        return f"AssistantMessage(content={self.text!r})"


class ToolResultMessage(_MessageBase):
    role: _MessageRole = _MessageRole.TOOL
    tool_call_id: str
    content: str

    def __str__(self) -> str:
        return f"ToolResultMessage(tool_call_id={self.tool_call_id}, content={_truncate(self.content)})"

    def __repr__(self) -> str:
        return f"ToolResultMessage(tool_call_id={self.tool_call_id!r}, content={_truncate(self.content)!r})"


type Message = UserMessage | SystemMessage | AssistantMessage | ToolResultMessage
