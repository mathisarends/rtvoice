from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """Wraps an agent skill — either from a SKILL.md file or inline content."""

    path: Path | None = None
    _name: str | None = field(default=None, init=False, repr=False)
    _instructions: str | None = field(default=None, init=False, repr=False)
    _summary: str | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        assert self.path is not None
        return self.path.name

    @property
    def instructions(self) -> str:
        if self._instructions is not None:
            return self._instructions
        assert self.path is not None
        skill_md = self.path / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"No SKILL.md found in {self.path}")
        return skill_md.read_text(encoding="utf-8")

    @property
    def summary(self) -> str:
        if self._summary is not None:
            return self._summary
        for line in self.instructions.splitlines():
            stripped = line.lstrip("#").strip()
            if stripped:
                return stripped
        return self.name

    @classmethod
    def from_dir(cls, path: Path | str) -> Self:
        if not Path(path).is_dir():
            raise ValueError(f"Skill path does not exist: {path}")
        return cls(path=Path(path))

    @classmethod
    def from_content(
        cls, name: str, instructions: str, summary: str | None = None
    ) -> Self:
        """Inline skill without filesystem — useful for tests and dynamically generated skills."""
        instance = cls()
        instance._name = name
        instance._instructions = instructions
        instance._summary = summary
        return instance

    @classmethod
    def discover(cls, directory: Path | str) -> list[Self]:
        """Recursively discover all skills in a directory."""
        root = Path(directory)
        return [cls(path=p.parent) for p in root.rglob("SKILL.md")]
