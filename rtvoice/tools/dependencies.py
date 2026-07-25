from typing import Protocol, runtime_checkable


@runtime_checkable
class Supervisor(Protocol):
    description: str
    handoff_instructions: str | None
    result_instructions: str | None

    async def start(self, task: str, context: str | None = None) -> str: ...
