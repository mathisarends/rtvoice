from dataclasses import dataclass, field

from rtvoice.llm import Message, ToolCall


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
    suppress_realtime_response: bool = False
    clarification_needed: str | None = None
    resume_history: list[Message] | None = None
    clarify_call_id: str | None = None

    def to_agent_output(self) -> str:
        return str(self)

    def __str__(self) -> str:
        parts: list[str] = []

        status = "✓" if self.success else "✗"
        parts.append(
            f"[{status}] {self.message or ('Success' if self.success else 'Failed')}"
        )

        if self.tool_calls:
            calls = ", ".join(tc.function.name for tc in self.tool_calls)
            parts.append(f"Tools used: {calls}")

        return " | ".join(parts)
