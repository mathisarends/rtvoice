from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    value: str | None = None
    error: str | None = None
    respond: bool | None = None
    # overrides the tool's static result_instruction for this call
    instruction: str | None = None

    @classmethod
    def success(
        cls,
        value: str | None = None,
        *,
        respond: bool | None = None,
        instruction: str | None = None,
    ) -> Self:
        return cls(ok=True, value=value, respond=respond, instruction=instruction)

    @classmethod
    def fail(cls, error: str | Exception, *, respond: bool | None = None) -> Self:
        return cls(ok=False, error=str(error), respond=respond)
