from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    value: Any = None
    error: str | None = None
    respond: bool | None = None

    @classmethod
    def success(cls, value: Any = None, *, respond: bool | None = None) -> Self:
        return cls(ok=True, value=value, respond=respond)

    @classmethod
    def fail(cls, error: str | Exception, *, respond: bool | None = None) -> Self:
        return cls(ok=False, error=str(error), respond=respond)
