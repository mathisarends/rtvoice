from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_FRONTMATTER_PATTERN = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)(.*)\Z",
    re.DOTALL,
)
_SKILL_NAME_PATTERN = re.compile(r"^(?!.*--)[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    directory: Path
    instructions: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] | None = None
    allowed_tools: str | None = None

    @property
    def location(self) -> Path:
        return self.directory / "SKILL.md"


def parse_skill(path: Path) -> Skill:
    resolved_path = path.resolve()
    try:
        source = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read skill file '{resolved_path}': {exc}") from exc

    match = _FRONTMATTER_PATTERN.match(source)
    if match is None:
        raise ValueError(
            f"Skill file '{resolved_path}' must start with YAML frontmatter."
        )

    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Skill file '{resolved_path}' has invalid YAML frontmatter: {exc}"
        ) from exc

    if not isinstance(frontmatter, dict):
        raise ValueError(f"Skill file '{resolved_path}' frontmatter must be a mapping.")

    name = _required_string(frontmatter, "name", resolved_path)
    description = _required_string(frontmatter, "description", resolved_path)
    instructions = match.group(2).strip()

    if not _SKILL_NAME_PATTERN.fullmatch(name) or len(name) > 64:
        raise ValueError(
            f"Skill name '{name}' must be 1-64 lowercase letters, numbers, or "
            "hyphens, without leading, trailing, or consecutive hyphens."
        )
    if name != resolved_path.parent.name:
        raise ValueError(
            f"Skill name '{name}' must match its parent directory "
            f"'{resolved_path.parent.name}'."
        )
    if len(description) > 1024:
        raise ValueError(f"Skill '{name}' description must be at most 1024 characters.")
    if not instructions:
        raise ValueError(f"Skill '{name}' must contain Markdown instructions.")

    license_name = _optional_string(frontmatter, "license", resolved_path)
    compatibility = _optional_string(frontmatter, "compatibility", resolved_path)
    if compatibility is not None and len(compatibility) > 500:
        raise ValueError(
            f"Skill '{name}' compatibility must be at most 500 characters."
        )

    allowed_tools = _optional_string(frontmatter, "allowed-tools", resolved_path)
    metadata = _metadata(frontmatter.get("metadata"), resolved_path)

    return Skill(
        name=name,
        description=description,
        directory=resolved_path.parent,
        instructions=instructions,
        license=license_name,
        compatibility=compatibility,
        metadata=metadata,
        allowed_tools=allowed_tools,
    )


def _required_string(frontmatter: dict[str, Any], key: str, path: Path) -> str:
    value = frontmatter.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Skill file '{path}' requires a non-empty string '{key}' field."
        )
    return value.strip()


def _optional_string(frontmatter: dict[str, Any], key: str, path: Path) -> str | None:
    value = frontmatter.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Skill file '{path}' field '{key}' must be a non-empty string."
        )
    return value.strip()


def _metadata(value: Any, path: Path) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(
            f"Skill file '{path}' field 'metadata' must map strings to strings."
        )
    return dict(value)
