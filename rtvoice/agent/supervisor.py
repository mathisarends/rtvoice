import asyncio
import json
import logging
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
from rtvoice.tools import Tools
from rtvoice.tools.di import ToolContext

logger = logging.getLogger(__name__)


@dataclass
class DoneSignal:
    result: str


@dataclass
class ClarifySignal:
    question: str


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
        self._llm = llm or ChatModel(model="gpt-5.4-mini")
        self._tools = Tools()
        if tools:
            self._tools.merge(tools)

        self._max_iterations = max_iterations
        self.handoff_instructions = handoff_instructions
        self.result_instructions = result_instructions
        self.holding_instruction = holding_instruction

        self._tools.set_context(ToolContext(context=context))
        self._pending_updates: asyncio.Queue[str] = asyncio.Queue()

        self._register_done_tool()
        self._register_clarify_tool()

    async def update(self, message: str) -> None:
        await self._pending_updates.put(message)

    def discard_pending_updates(self) -> None:
        while not self._pending_updates.empty():
            self._pending_updates.get_nowait()
            self._pending_updates.task_done()

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

    async def start(
        self,
        task: str,
        context: str | None = None,
    ) -> SupervisorResult:
        messages = self._build_messages(task=task, context=context)
        return await self._loop(messages)

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

    async def resume(
        self,
        clarification_answer: str,
        resume_history: list[Message],
        clarify_call_id: str,
    ) -> SupervisorResult:
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
            self._append_pending_updates(messages)
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
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Failed to parse arguments for tool '%s': %s", tool_name, exc
                    )
                    messages.append(
                        ToolResultMessage(
                            tool_call_id=tool_call.id,
                            content=f"Error: could not parse tool arguments – {exc}. Please retry with valid JSON.",
                        )
                    )
                    continue

                logger.debug("Executing supervisor tool call: '%s'", tool_name)
                try:
                    result = await self._tools.execute(tool_name, tool_args)
                except Exception as exc:
                    logger.warning(
                        "Tool '%s' raised an error: %s", tool_name, exc, exc_info=True
                    )
                    messages.append(
                        ToolResultMessage(
                            tool_call_id=tool_call.id,
                            content=f"Error: tool '{tool_name}' failed with: {exc}. Please handle this and try again.",
                        )
                    )
                    continue

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
                messages.append(
                    ToolResultMessage(tool_call_id=tool_call.id, content=str(result))
                )

        return SupervisorDone(
            message="Max iterations reached.",
            success=False,
        )

    def _append_pending_updates(self, messages: list[Message]) -> None:
        while not self._pending_updates.empty():
            update = self._pending_updates.get_nowait()
            messages.append(
                UserMessage(
                    content=f"<supervisor_update>\n{update}\n</supervisor_update>"
                )
            )
            self._pending_updates.task_done()
