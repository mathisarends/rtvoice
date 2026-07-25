from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any, Self

from pydantic import BaseModel

from rtvoice.realtime.schemas import FunctionTool
from rtvoice.skills.bash import BashRunner
from rtvoice.skills.manager import SkillManager
from rtvoice.tools.binding import ToolAvailability, ToolDescription
from rtvoice.tools.di import ToolContext
from rtvoice.tools.executor import ToolExecutor
from rtvoice.tools.params import BashParams, LoadSkillParams
from rtvoice.tools.results import ActionResult
from rtvoice.tools.views import ActionKind, Tool

logger = logging.getLogger(__name__)


class Tools:
    def __init__(
        self,
        *,
        skill_manager: SkillManager | None = None,
        allowed_commands: Iterable[str] = (),
    ):
        self.tools: dict[str, Tool] = {}
        self._context: ToolContext | None = None
        self._executor = ToolExecutor(self.tools, self._context)
        self._register_default_tools(skill_manager, allowed_commands)

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
        self._executor.set_context(context)

    def inject_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def eject_tool(self, name: str) -> None:
        self.tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def get_tool_schema(self) -> list[FunctionTool]:
        return [
            tool.to_schema(self._context)
            for tool in self.tools.values()
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
        return await self._executor.execute(name, arguments)

    def clone(self) -> Self:
        new = type(self)()
        # mutate in place so the clone's executor keeps referencing this dict
        new.tools.update(self.tools)
        return new

    def merge(self, other: Tools) -> None:
        for tool in other.tools.values():
            self._register_tool(tool)

    def is_registered(self, tool: Tool) -> bool:
        return tool in self.tools.values()

    def _register_tool(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self.tools[tool.name] = tool

    def _register_default_tools(
        self,
        skill_manager: SkillManager | None,
        allowed_commands: Iterable[str],
    ) -> None:
        if skill_manager is None or skill_manager.size == 0:
            return

        bash_runner = BashRunner(
            allowed_commands=tuple(dict.fromkeys(allowed_commands)),
            allowed_script_dirs=skill_manager.directories,
        )

        @self.action(
            "Load a skill's full instructions or one bundled resource. Call this "
            "before using a skill. Pass only a relative resource path.",
            name="load_skill",
            params=LoadSkillParams,
        )
        def load_skill(params: LoadSkillParams) -> ActionResult:
            return ActionResult.success(skill_manager.load(params.name, params.path))

        @self.action(
            "Execute a Bash command. Only explicitly allowed commands or "
            "scripts inside an available skill directory may run.",
            name="bash",
            params=BashParams,
            kind=ActionKind.DESTRUCTIVE,
        )
        async def bash(params: BashParams) -> ActionResult:
            return ActionResult.success(await bash_runner.execute(params))
