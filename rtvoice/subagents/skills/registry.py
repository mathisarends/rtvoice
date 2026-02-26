import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class SkillRegistry:
    def __init__(self, skills_dir: Path):
        self._dir = skills_dir
        self._index: dict[str, str] = self._build_index()

    def _build_index(self) -> dict[str, str]:
        index = {}
        for skill_md in self._dir.glob("*/SKILL.md"):
            try:
                content = skill_md.read_text()
                name = skill_md.parent.name
                index[name] = self._extract_description(content)
            except Exception as e:
                logger.warning("Could not index skill at %s: %s", skill_md, e)
        return index

    def _extract_description(self, content: str) -> str:
        match = re.search(r"^description:\s*(.+)$", content, re.MULTILINE)
        return match.group(1).strip() if match else "No description."

    def load(self, name: str) -> str:
        skill_path = self._dir / name / "SKILL.md"
        if not skill_path.exists():
            return f"Skill '{name}' not found."
        return skill_path.read_text()

    def load_resource(self, skill_name: str, resource_path: str) -> str:
        full_path = self._dir / skill_name / resource_path
        if not full_path.resolve().is_relative_to(self._dir.resolve()):
            return "Access denied: path outside skills directory."
        if not full_path.exists():
            return f"Resource '{resource_path}' not found in skill '{skill_name}'."
        return full_path.read_text()

    def as_prompt_section(self) -> str:
        if not self._index:
            return ""
        skills_list = "\n".join(
            f"- {name}: {desc}" for name, desc in self._index.items()
        )
        return (
            "## Available Skills\n"
            "Call `load_skill` with the skill name before starting a relevant task.\n"
            f"{skills_list}"
        )

    def is_empty(self) -> bool:
        return not self._index
