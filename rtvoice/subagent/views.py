from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from rtvoice.llm import Message
from rtvoice.token.views import TokenUsageSummary


@dataclass
class DoneSignal:
    result: str


@dataclass
class ClarifySignal:
    question: str


@dataclass
class ProgressSignal:
    message: str


type ToolSignal = DoneSignal | ClarifySignal | ProgressSignal

type ProgressCallback = Callable[[str], Awaitable[None]]


class AgentDone(BaseModel):
    """The agent completed the task successfully."""

    message: str
    success: bool = True
    token_usage: TokenUsageSummary = Field(default_factory=TokenUsageSummary)


class AgentClarificationNeeded(BaseModel):
    """The agent cannot proceed without an answer from the user."""

    question: str
    resume_history: list[Message]
    clarify_call_id: str
    token_usage: TokenUsageSummary = Field(default_factory=TokenUsageSummary)


type SubAgentResult = AgentDone | AgentClarificationNeeded
