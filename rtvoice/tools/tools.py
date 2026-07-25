from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Self

from pydantic import BaseModel

from rtvoice.agent.subagent import Subagent
from rtvoice.conversation import ConversationHistory
from rtvoice.realtime.schemas import FunctionTool
from rtvoice.skills.bash import BashRunner
from rtvoice.skills.manager import SkillManager
from rtvoice.tools.argument_resolver import resolve_arguments
from rtvoice.tools.binding import (
    ToolAvailability,
    ToolDescription,
    described,
    provided,
    requires,
)
from rtvoice.tools.di import Inject, ToolContext
from rtvoice.tools.middleware import MiddlewareChain, ToolCall
from rtvoice.tools.params import BashParams, LoadSkillParams, SubagentParams
from rtvoice.tools.results import ActionResult
from rtvoice.tools.views import ActionKind, Tool

logger = logging.getLogger(__name__)


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
                    status=status,
                    kind=kind,
                    available_when=available_when,
                )
            )
            return func

        return decorator

    def set_context(self, context: ToolContext) -> None:
        self._context = context
        subagent = context.resolve(Subagent)
        self._tools["subagent"].result_instruction = (
            subagent.result_instructions if subagent is not None else None
        )

    def inject_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def eject_tool(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_tool_schema(self) -> list[FunctionTool]:
        return [
            tool.to_schema(self._context)
            for tool in self._tools.values()
            if tool.is_available(self._context)
        ]

    def get_json_tool_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": tool.model_dump(exclude={"type"}, exclude_none=True),
            }
            for tool in self.get_tool_schema()
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

    def clone(self) -> Self:
        new = type(self)()
        # mutate in place so the clone's handler chain keeps referencing this dict
        new._tools.update(self._tools)
        return new

    def merge(self, other: Tools) -> None:
        for tool in other._tools.values():
            # every Tools carries the defaults, so merging must not collide on them
            if tool.name in self._default_tool_names:
                continue
            self._register_tool(tool)

    def is_registered(self, tool: Tool) -> bool:
        return tool in self._tools.values()

    def _register_tool(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def _register_default_tools(self) -> None:
        @self.action(
            described(
                Subagent,
                render=_describe_subagent,
                default="Delegate a task to the subagent.",
            ),
            name="subagent",
            params=SubagentParams,
            available_when=provided(Subagent),
        )
        async def subagent(
            params: SubagentParams,
            subagent: Inject[Subagent],
            conversation_history: Inject[ConversationHistory],
        ) -> str:
            conversation_summary = conversation_history.format_summary()
            return await subagent.start(
                params.task,
                context=conversation_summary,
            )

        @self.action(
            "Load a skill's full instructions or one bundled resource. Call this "
            "before using a skill. Pass only a relative resource path.",
            params=LoadSkillParams,
            available_when=requires(
                SkillManager, predicate=lambda manager: manager.size > 0
            ),
        )
        def load_skill(
            params: LoadSkillParams, skill_manager: Inject[SkillManager]
        ) -> ActionResult:
            content = skill_manager.load(params.name, params.path)
            return ActionResult.success(content)

        @self.action(
            "Execute a Bash command. Only explicitly allowed commands or "
            "scripts inside an available skill directory may run.",
            params=BashParams,
            kind=ActionKind.DESTRUCTIVE,
            available_when=provided(BashRunner),
        )
        async def bash(
            params: BashParams, bash_runner: Inject[BashRunner]
        ) -> ActionResult:
            output = await bash_runner.execute(params)
            return ActionResult.success(output)


def _describe_subagent(subagent: Subagent) -> str:
    if not subagent.handoff_instructions:
        return subagent.description
    return (
        f"{subagent.description}\n\n"
        f"Handoff instructions: {subagent.handoff_instructions}"
    )
