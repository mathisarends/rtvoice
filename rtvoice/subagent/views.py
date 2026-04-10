from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel

from rtvoice.llm import Message


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


class AgentClarificationNeeded(BaseModel):
    """The agent cannot proceed without an answer from the user."""

    question: str
    resume_history: list[Message]
    clarify_call_id: str


type SubAgentResult = AgentDone | AgentClarificationNeeded
