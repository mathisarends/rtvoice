from collections.abc import Iterable
from html import escape
from typing import Protocol


class _SkillInfo(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...


class SystemPrompt:
    def __init__(self, content: str, *, skills: Iterable[_SkillInfo] = ()) -> None:
        self._content = _append_skill_discovery(content, skills)

    def __str__(self) -> str:
        return self._content


def _append_skill_discovery(
    content: str,
    skills: Iterable[_SkillInfo],
) -> str:
    entries = "\n".join(
        (
            "<skill>\n"
            f"<name>{escape(skill.name)}</name>\n"
            f"<description>{escape(skill.description)}</description>\n"
            "</skill>"
        )
        for skill in skills
    )
    if not entries:
        return content

    discovery = (
        "## Available Skills\n\n"
        "<usage>\n"
        "Skills provide specialized capabilities and domain knowledge. "
        "Use them when they match the current task.\n\n"
        'Load a skill with `load_skill(name="<skill-name>")` before '
        "following its instructions. It lists the skill's bundled files. "
        "Read one with `read_skill_resource`, run one with "
        "`run_skill_script`. Both take paths relative to the skill.\n"
        "</usage>\n\n"
        "<available_skills>\n"
        f"{entries}\n"
        "</available_skills>"
    )
    if not content:
        return discovery
    return f"{content.rstrip()}\n\n{discovery}"
