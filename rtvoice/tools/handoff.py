from abc import ABC, abstractmethod


class Handoff(ABC):
    """Contract for an agent the realtime agent can delegate a task to. Lives
    in the tools package so the handoff tool can be a default tool without
    tools importing the agent that implements it."""

    description: str
    handoff_instructions: str | None
    result_instructions: str | None

    @abstractmethod
    async def start(self, task: str, context: str | None = None) -> str: ...
