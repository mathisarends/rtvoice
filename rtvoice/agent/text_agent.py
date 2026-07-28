from __future__ import annotations

import json
import logging

from rtvoice.agent.system_prompt import SystemPrompt
from rtvoice.llm import (
    AssistantMessage,
    ChatModel,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from rtvoice.skills import Skills
from rtvoice.tools import Handoff, ToolContext, Tools, ToolSchemaFormat

logger = logging.getLogger(__name__)


class TextAgent[T](Handoff):
    def __init__(
        self,
        *,
        description: str,
        system_prompt: str,
        llm: ChatModel | None = None,
        tools: Tools | None = None,
        skills: Skills | None = None,
        max_iterations: int = 10,
        handoff_instructions: str | None = None,
        result_instructions: str | None = None,
        tool_injection_context: T | None = None,
    ) -> None:
        self.name = "text_agent"
        self.description = description
        self._llm = llm or ChatModel(model="gpt-5.4-mini")
        self._skills = skills
        self._tools = Tools()
        if tools:
            self._tools.merge(tools)

        self._system_prompt = SystemPrompt(
            system_prompt,
            skills=self._skills if self._skills is not None else (),
        )

        self._max_iterations = max_iterations
        self.handoff_instructions = handoff_instructions
        self.result_instructions = result_instructions

        # withholding the Handoff keeps the handoff tool unavailable here: a
        # delegated agent must not delegate to another agent again
        self._tools.set_context(
            ToolContext(tool_injection_context, self._skills).without(Handoff)
        )

    async def start(
        self,
        task: str,
        context: str | None = None,
    ) -> str:
        messages = self._build_messages(task=task, context=context)
        return await self._loop(messages)

    def _build_messages(self, task: str, context: str | None) -> list[Message]:
        messages = [SystemMessage(content=str(self._system_prompt))]
        if context:
            messages.append(
                UserMessage(
                    content=f"<conversation_history>\n{context}\n</conversation_history>"
                )
            )
        messages.append(UserMessage(content=f"<task>\n{task}\n</task>"))
        return messages

    async def _loop(self, messages: list[Message]) -> str:
        tool_schema = self._tools.get_schema(ToolSchemaFormat.TEXT)

        for _ in range(self._max_iterations):
            response = await self._llm.invoke(messages, tools=tool_schema)

            if not response.tool_calls:
                return response.completion

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

                result = await self._tools.execute(tool_name, tool_args)
                if not result.ok:
                    messages.append(
                        ToolResultMessage(
                            tool_call_id=tool_call.id,
                            content=(
                                f"Error: tool '{tool_name}' failed with: {result.error}. "
                                "Please handle this and try again."
                            ),
                        )
                    )
                    continue

                content = "OK" if result.value is None else str(result.value)
                messages.append(
                    ToolResultMessage(tool_call_id=tool_call.id, content=content)
                )

        return "Max iterations reached."
