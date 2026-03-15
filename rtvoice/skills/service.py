from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Annotated, Self

import yaml
from typing_extensions import Doc

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """Wraps an agent skill — either from a SKILL.md file or inline content."""

    path: Annotated[
        Path | None,
        Doc("Path to the skill directory containing a SKILL.md file."),
    ] = None
    _name: str | None = field(default=None, init=False, repr=False)
    _instructions: str | None = field(default=None, init=False, repr=False)
    _summary: str | None = field(default=None, init=False, repr=False)

    @property
    def name(
        self,
    ) -> Annotated[str, Doc("Skill name from frontmatter or inline content.")]:
        return self._name or self._frontmatter["name"]

    @property
    def summary(
        self,
    ) -> Annotated[
        str,
        Doc(
            "Short description from frontmatter or inline content. Used in skill index."
        ),
    ]:
        return self._summary or self._frontmatter["description"]

    @property
    def instructions(
        self,
    ) -> Annotated[
        str, Doc("Full skill instructions — SKILL.md body without frontmatter.")
    ]:
        if self._instructions is not None:
            return self._instructions
        return self._strip_frontmatter(self._read_raw())

    @classmethod
    def from_dir(
        cls,
        path: Annotated[
            Path | str,
            Doc(
                "Path to a skill directory containing a SKILL.md file with valid "
                "YAML frontmatter. The frontmatter `name` field must match the "
                "directory name."
            ),
        ],
    ) -> Self:
        """Load a skill from a directory on disk.

        Validates that the directory exists and the SKILL.md frontmatter
        contains the required `name` and `description` fields.

        Raises:
            ValueError: If the path does not exist, frontmatter is missing or
                malformed, or the `name` field does not match the directory name.
            FileNotFoundError: If no SKILL.md file is found in the directory.
        """
        resolved = Path(path)
        if not resolved.is_dir():
            raise ValueError(f"Skill path does not exist: {resolved}")
        instance = cls(path=resolved)
        instance._validate()
        return instance

    @classmethod
    def from_content(
        cls,
        name: Annotated[str, Doc("Unique skill identifier.")],
        instructions: Annotated[str, Doc("Full skill instructions in Markdown.")],
        summary: Annotated[
            str | None,
            Doc(
                "Short description shown in the skill index. "
                "Defaults to the first non-empty line of `instructions`."
            ),
        ] = None,
    ) -> Self:
        """Create an inline skill without a filesystem path.

        Useful for tests, dynamically generated skills, or skills embedded
        directly in code. Bypasses frontmatter parsing entirely.
        """
        instance = cls()
        instance._name = name
        instance._instructions = instructions
        instance._summary = summary
        return instance

    @classmethod
    def discover(
        cls,
        directory: Annotated[
            Path | str,
            Doc(
                "Root directory to search. All subdirectories containing a "
                "SKILL.md file are loaded as skills."
            ),
        ],
    ) -> list[Self]:
        """Recursively discover all skills in a directory.

        Each subdirectory containing a SKILL.md file is loaded via `from_dir`,
        including full frontmatter validation.
        """
        root = Path(directory)
        return [cls.from_dir(p.parent) for p in root.rglob("SKILL.md")]

    @cached_property
    def _frontmatter(self) -> dict:
        raw = self._read_raw()
        if not raw.startswith("---"):
            raise ValueError(f"SKILL.md in '{self.path}' is missing YAML frontmatter.")
        parts = raw.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"SKILL.md in '{self.path}' has malformed frontmatter.")
        return yaml.safe_load(parts[1]) or {}

    def _read_raw(self) -> str:
        assert self.path is not None
        skill_md = self.path / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"No SKILL.md found in {self.path}")
        return skill_md.read_text(encoding="utf-8")

    @staticmethod
    def _strip_frontmatter(raw: str) -> str:
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return raw.strip()

    def _validate(self) -> None:
        assert self.path is not None
        fm = self._frontmatter
        missing = [f for f in ("name", "description") if not fm.get(f)]
        if missing:
            raise ValueError(
                f"SKILL.md in '{self.path}' missing required frontmatter fields: {missing}"
            )
        if fm["name"] != self.path.name:
            raise ValueError(
                f"Frontmatter 'name' ('{fm['name']}') must match directory name ('{self.path.name}')."
            )
