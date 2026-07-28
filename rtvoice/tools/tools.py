from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal, overload

from pydantic import BaseModel
from transitbus import EventBus

from rtvoice.conversation import ConversationHistory
from rtvoice.events.views import StopAgentCommand
from rtvoice.llm import RawSchemaTool
from rtvoice.realtime.schemas import FunctionTool as RealtimeFunctionTool
from rtvoice.skills import Skills
from rtvoice.tools.argument_resolver import resolve_arguments
from rtvoice.tools.binding import (
    ToolAvailability,
    ToolDescription,
    described,
    provided,
    requires,
)
from rtvoice.tools.di import Inject, ToolContext
from rtvoice.tools.handoff import Handoff
from rtvoice.tools.middleware import MiddlewareChain, ToolCall
from rtvoice.tools.params import (
    LoadSkillParams,
    ReadSkillResourceParams,
    RunSkillScriptParams,
    TextAgentParams,
)
from rtvoice.tools.results import ActionResult
from rtvoice.tools.views import ActionKind, Tool

logger = logging.getLogger(__name__)


def _describe_handoff(handoff: Handoff) -> str:
    if not handoff.handoff_instructions:
        return handoff.description
    return (
        f"{handoff.description}\n\nHandoff instructions: {handoff.handoff_instructions}"
    )


class ToolSchemaFormat(StrEnum):
    REALTIME = "realtime"
    TEXT = "text"


class Tools:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._context: ToolContext | None = None
        self._handler = MiddlewareChain(self._tools).build(self._invoke)
        self._register_default_tools()
        self._default_tool_names = frozenset(self._tools)

    def action(
        self,
        description: str | ToolDescription,
        name: str | None = None,
        *,
        params: type[BaseModel] | None = None,
        result_instruction: str | None = None,
        respond: bool = True,
        status: str | Callable | None = None,
        kind: ActionKind = ActionKind.GENERIC,
        available_when: ToolAvailability | None = None,
    ) -> Callable:
        def decorator(func: Callable) -> Callable:
            self._register_tool(
                Tool(
                    name=name or func.__name__,
                    description=description,
                    fn=func,
                    param_model=params,
                    result_instruction=result_instruction,
                    respond=respond,
                    status=status,
                    kind=kind,
                    available_when=available_when,
                )
            )
            return func

        return decorator

    def set_context(self, context: ToolContext) -> None:
        self._context = context

    def inject_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @overload
    def get_schema(
        self, schema_format: Literal[ToolSchemaFormat.REALTIME] = ...
    ) -> list[RealtimeFunctionTool]: ...

    @overload
    def get_schema(
        self, schema_format: Literal[ToolSchemaFormat.TEXT]
    ) -> list[RawSchemaTool]: ...

    def get_schema(
        self, schema_format: ToolSchemaFormat = ToolSchemaFormat.REALTIME
    ) -> list[RealtimeFunctionTool] | list[RawSchemaTool]:
        schemas = [
            tool.to_schema(self._context)
            for tool in self._tools.values()
            if tool.is_available(self._context)
        ]
        if schema_format is ToolSchemaFormat.REALTIME:
            return schemas
        return [
            RawSchemaTool(
                name=schema.name,
                description=schema.description or "",
                schema=schema.parameters.model_dump(exclude_none=True),
            )
            for schema in schemas
        ]

    async def execute(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ActionResult:
        return await self._handler(
            ToolCall(name=name, raw_args=arguments or {}, context=self._context)
        )

    async def _invoke(self, call: ToolCall) -> ActionResult:
        resolved_args = resolve_arguments(
            call.tool, call.raw_args, call.params, call.context
        )
        result = await call.tool.execute(resolved_args)
        if isinstance(result, ActionResult):
            return result
        return ActionResult.success(result)

    def merge(self, other: Tools) -> None:
        for tool in other._tools.values():
            # every Tools carries the defaults, so merging must not collide on them
            if tool.name in self._default_tool_names:
                continue
            self._register_tool(tool)

    def _register_tool(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def _register_default_tools(self) -> None:
        """Every Tools instance carries these; `available_when` decides which of
        them a given agent actually sees, based on what its context provides."""
        with_skills = requires(Skills, predicate=lambda skills: skills.size > 0)

        @self.action(
            "End the conversation and shut the agent down. Call this when the user "
            "says goodbye or asks you to stop. Say a short farewell first.",
            name="stop",
            kind=ActionKind.END_SESSION,
            available_when=provided(EventBus),
        )
        async def _stop(event_bus: Inject[EventBus]) -> ActionResult:
            await event_bus.dispatch(StopAgentCommand())
            return ActionResult.success("Conversation ended.")

        @self.action(
            "Load a skill's instructions and the list of its bundled files. "
            "Call this before using a skill.",
            params=LoadSkillParams,
            available_when=with_skills,
        )
        def load_skill(params: LoadSkillParams, skills: Inject[Skills]) -> ActionResult:
            return ActionResult.success(skills.load(params.name))

        @self.action(
            "Read one file bundled with a skill, as listed by load_skill.",
            params=ReadSkillResourceParams,
            kind=ActionKind.READ,
            available_when=with_skills,
        )
        def read_skill_resource(
            params: ReadSkillResourceParams, skills: Inject[Skills]
        ) -> ActionResult:
            return ActionResult.success(skills.read_resource(params.name, params.path))

        @self.action(
            "Run one script bundled with a skill, as listed by load_skill. The "
            "script runs in the skill's directory; no shell is involved.",
            params=RunSkillScriptParams,
            kind=ActionKind.DESTRUCTIVE,
            available_when=with_skills,
        )
        async def run_skill_script(
            params: RunSkillScriptParams, skills: Inject[Skills]
        ) -> ActionResult:
            output = await skills.run_script(
                params.name, params.path, params.args, params.timeout
            )
            return ActionResult.success(output)

        @self.action(
            described(
                Handoff,
                render=_describe_handoff,
                default="Delegate a task to the text agent.",
            ),
            params=TextAgentParams,
            available_when=provided(Handoff),
        )
        async def text_agent(
            params: TextAgentParams,
            handoff: Inject[Handoff],
            conversation_history: Inject[ConversationHistory],
        ) -> ActionResult:
            answer = await handoff.start(
                params.task, context=conversation_history.format()
            )
            return ActionResult.success(answer, instruction=handoff.result_instructions)
