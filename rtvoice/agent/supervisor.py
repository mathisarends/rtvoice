from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from rtvoice.agent.views import (
    SupervisorClarificationNeeded,
    SupervisorDone,
    SupervisorResult,
)
from rtvoice.llm import (
    AssistantMessage,
    ChatModel,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from rtvoice.shared.decorators import timed
from rtvoice.tools import Tools
from rtvoice.tools.di import ToolContext

logger = logging.getLogger(__name__)


@dataclass
class DoneSignal:
    result: str


@dataclass
class ClarifySignal:
    question: str


@dataclass
class ProgressSignal:
    message: str


type ProgressCallback = Callable[[str], Awaitable[None]]


class Supervisor[T]:
    def __init__(
        self,
        *,
        description: str,
        instructions: str,
        llm: ChatModel | None = None,
        tools: Tools | None = None,
        max_iterations: int = 10,
        handoff_instructions: str | None = None,
        result_instructions: str | None = None,
        holding_instruction: str | None = None,
        context: T | None = None,
    ) -> None:
        self.name = "supervisor"
        self.description = description
        self._instructions = instructions
        self._llm = llm
        self._tools = Tools()
        if tools:
            self._tools.merge(tools)

        self._max_iterations = max_iterations
        self.handoff_instructions = handoff_instructions
        self.result_instructions = result_instructions
        self.holding_instruction = holding_instruction

        self._tools.set_context(ToolContext(context=context))

        self.on_progress: ProgressCallback | None = None
        self._on_progress: ProgressCallback | None = None

        self._register_done_tool()
        self._register_clarify_tool()
        self._register_progress_tool()

    def _register_done_tool(self) -> None:
        @self._tools.action(
            "Signal that the task is complete and return the final result to the user. "
            "Only call this once you have gathered all necessary information or took the appropriate action."
        )
        def done(result: str) -> DoneSignal:
            return DoneSignal(result)

    def _register_clarify_tool(self) -> None:
        @self._tools.action(
            "Ask the user a clarifying question when essential information is missing. "
            "Use sparingly - only when you cannot proceed without the answer. "
            "Calling this tool immediately returns control to the user; "
            "you will be called again once they answer."
        )
        def clarify(question: str) -> ClarifySignal:
            return ClarifySignal(question)

    def _register_progress_tool(self) -> None:
        @self._tools.action(
            "Report an intermediate progress update to the user while working on a long-running task. "
            "Use this to keep the user informed without blocking - the loop continues immediately after."
        )
        def report_progress(message: str) -> ProgressSignal:
            return ProgressSignal(message)

    @timed()
    async def prewarm(self) -> None:
        pass

    async def run(
        self,
        task: str,
        context: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> SupervisorResult:
        self._on_progress = on_progress or self.on_progress
        await self.prewarm()
        messages = self._build_messages(task=task, context=context)
        return await self._loop(messages)

    @timed()
    async def resume(
        self,
        clarification_answer: str,
        resume_history: list[Message],
        clarify_call_id: str,
        on_progress: ProgressCallback | None = None,
    ) -> SupervisorResult:
        self._on_progress = on_progress or self.on_progress
        messages = list(resume_history)
        messages.append(
            ToolResultMessage(
                tool_call_id=clarify_call_id,
                content=clarification_answer,
            )
        )
        return await self._loop(messages)

    async def _loop(self, messages: list[Message]) -> SupervisorResult:
        tool_schema = self._tools.get_json_tool_schema()

        for _ in range(self._max_iterations):
            response = await self._llm.invoke(messages, tools=tool_schema)

            if not response.tool_calls:
                return SupervisorDone(message=response.completion)

            messages.append(
                AssistantMessage(
                    content=response.completion,
                    tool_calls=response.tool_calls,
                )
            )

            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                logger.debug("Executing supervisor tool call: '%s'", tool_name)
                result = await self._tools.execute(tool_name, tool_args)

                match result:
                    case DoneSignal(result=message):
                        logger.debug(
                            "Supervisor tool 'done' called with result: %s", message
                        )
                        return SupervisorDone(message=message)
                    case ClarifySignal(question=question):
                        logger.debug(
                            "Supervisor tool 'clarify' called with question: %s",
                            question,
                        )
                        return SupervisorClarificationNeeded(
                            question=question,
                            resume_history=list(messages),
                            clarify_call_id=tool_call.id,
                        )
                    case ProgressSignal(message=message):
                        logger.debug("Supervisor progress update: %s", message)
                        if self._on_progress:
                            await self._on_progress(message)
                        messages.append(
                            ToolResultMessage(
                                tool_call_id=tool_call.id,
                                content="Progress noted, continue.",
                            )
                        )
                        continue

                messages.append(
                    ToolResultMessage(tool_call_id=tool_call.id, content=str(result))
                )

        return SupervisorDone(
            message="Max iterations reached.",
            success=False,
        )

    def _build_messages(self, task: str, context: str | None) -> list[Message]:
        messages = [SystemMessage(content=self._instructions)]
        if context:
            messages.append(
                UserMessage(
                    content=f"<conversation_history>\n{context}\n</conversation_history>"
                )
            )
        messages.append(UserMessage(content=f"<task>\n{task}\n</task>"))
        return messages
