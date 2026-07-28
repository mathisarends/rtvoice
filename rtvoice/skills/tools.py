from rtvoice.skills.collection import Skills
from rtvoice.tools.binding import ToolAvailability, requires
from rtvoice.tools.di import Inject
from rtvoice.tools.params import (
    LoadSkillParams,
    ReadSkillResourceParams,
    RunSkillScriptParams,
)
from rtvoice.tools.results import ActionResult
from rtvoice.tools.tools import Tools
from rtvoice.tools.views import ActionKind


def register_skill_tools(tools: Tools) -> None:
    @tools.action(
        "Load a skill's instructions and the list of its bundled files. "
        "Call this before using a skill.",
        params=LoadSkillParams,
        available_when=_with_skills(),
    )
    def load_skill(params: LoadSkillParams, skills: Inject[Skills]) -> ActionResult:
        return ActionResult.success(skills.load(params.name))

    @tools.action(
        "Read one file bundled with a skill, as listed by load_skill.",
        params=ReadSkillResourceParams,
        kind=ActionKind.READ,
        available_when=_with_skills(),
    )
    def read_skill_resource(
        params: ReadSkillResourceParams, skills: Inject[Skills]
    ) -> ActionResult:
        content = skills.read_resource(params.name, params.path)
        return ActionResult.success(content)

    @tools.action(
        "Run one script bundled with a skill, as listed by load_skill. The "
        "script runs in the skill's directory; no shell is involved.",
        params=RunSkillScriptParams,
        kind=ActionKind.DESTRUCTIVE,
        available_when=_with_skills(),
    )
    async def run_skill_script(
        params: RunSkillScriptParams, skills: Inject[Skills]
    ) -> ActionResult:
        output = await skills.run_script(
            params.name, params.path, params.args, params.timeout
        )
        return ActionResult.success(output)


def _with_skills() -> ToolAvailability:
    return requires(Skills, predicate=lambda skills: skills.size > 0)
