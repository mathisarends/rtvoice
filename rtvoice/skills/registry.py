from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from rtvoice.skills.service import Skill

logger = logging.getLogger(__name__)


@dataclass
class SkillRegistry:
    skills: list[Skill] = field(default_factory=list)

    def add(self, *skills: Skill | Path | str) -> None:
        for s in skills:
            if isinstance(s, Skill):
                self.skills.append(s)
            else:
                self.skills.append(Skill.from_dir(s))

    def get(self, name: str) -> Skill | None:
        return next((s for s in self.skills if s.name == name), None)

    def available_names(self) -> list[str]:
        return [s.name for s in self.skills]

    def format_index(self) -> str:
        """Short index injected into system prompt so the agent knows what's available."""
        lines = ["<available_skills>"]
        for skill in self.skills:
            lines.append(f"  <skill name='{skill.name}'>{skill.summary}</skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def format_all_instructions(self) -> str:
        """Inject all skill instructions at once (for smaller skill sets)."""
        blocks = []
        for skill in self.skills:
            blocks.append(
                f"<skill name='{skill.name}'>\n{skill.instructions}\n</skill>"
            )
        return "<skills>\n" + "\n\n".join(blocks) + "\n</skills>"
