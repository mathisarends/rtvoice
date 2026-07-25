from __future__ import annotations

from collections.abc import Iterable

from rtvoice.skills.bash import BashArgs, BashRunner
from rtvoice.skills.manager import SkillManager, Skills
from rtvoice.tools import Tools


class SkillRuntime:
    def __init__(
        self,
        config: Skills,
        *,
        allowed_commands: Iterable[str] = (),
    ) -> None:
        self._manager = SkillManager(config)
        self._bash = BashRunner(
            allowed_commands=tuple(dict.fromkeys(allowed_commands)),
            allowed_script_dirs=self._manager.directories,
        )

    @property
    def enabled(self) -> bool:
        return self._manager.size > 0

    @property
    def discovery_prompt(self) -> str:
        return self._manager.discovery_prompt()

    def register_tools(self, tools: Tools) -> None:
        if not self.enabled:
            return

        @tools.action(
            "Load a skill's full instructions or one bundled resource. Call this "
            "before using a skill. Pass only a relative resource path.",
            name="load_skill",
        )
        def load_skill(name: str, path: str | None = "SKILL.md") -> str:
            return self._manager.load(name, path)

        @tools.action(
            "Execute a Bash command. Only explicitly allowed commands or "
            "scripts inside an available skill directory may run.",
            name="bash",
            param_model=BashArgs,
        )
        async def bash(args: BashArgs) -> str:
            return await self._bash.execute(args)


def append_skill_prompt(instructions: str, runtime: SkillRuntime | None) -> str:
    if runtime is None or not runtime.enabled:
        return instructions
    if not instructions:
        return runtime.discovery_prompt
    return f"{instructions.rstrip()}\n\n{runtime.discovery_prompt}"
