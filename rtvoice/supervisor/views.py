import asyncio
from dataclasses import dataclass, field

from llmify import ToolCall


class SupervisorAgentDone(Exception):
    def __init__(self, result: str):
        self.result = result


@dataclass
class SupervisorAgentClarificationNeeded(Exception):
    question: str
    answer_future: asyncio.Future


@dataclass
class SupervisorAgentResult:
    message: str
    success: bool = True
    tool_calls: list[ToolCall] = field(default_factory=list)

    def __str__(self) -> str:
        return self.message or ("Success" if self.success else "Failed")
