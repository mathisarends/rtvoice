import asyncio
from dataclasses import dataclass

from pydantic import BaseModel


class SubAgentDone(Exception):
    def __init__(self, result: str):
        self.result = result


@dataclass
class SubAgentClarificationNeeded(Exception):
    question: str
    answer_future: asyncio.Future


class ToolCall(BaseModel):
    name: str
    arguments: dict
    result: str


class SubAgentResult(BaseModel):
    success: bool = True
    message: str | None = None
    tool_calls: list[ToolCall] = []

    def __str__(self) -> str:
        return self.message or ("Success" if self.success else "Failed")
