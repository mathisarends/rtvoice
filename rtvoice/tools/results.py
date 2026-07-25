from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    value: Any = None
    error: str | None = None

    @classmethod
    def success(cls, value: Any = None) -> ActionResult:
        return cls(ok=True, value=value)

    @classmethod
    def fail(cls, error: str | Exception) -> ActionResult:
        return cls(ok=False, error=str(error))
