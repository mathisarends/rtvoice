from dataclasses import dataclass, field

from llmify import Message, ToolCall


@dataclass
class DoneSignal:
    result: str


@dataclass
class ClarifySignal:
    question: str


type ToolSignal = DoneSignal | ClarifySignal


@dataclass
class SubAgentResult:
    message: str
    success: bool = True
    tool_calls: list[ToolCall] = field(default_factory=list)
    clarification_needed: str | None = None
    resume_history: list[Message] | None = None
    clarify_call_id: str | None = None

    def __str__(self) -> str:
        return self.message or ("Success" if self.success else "Failed")
