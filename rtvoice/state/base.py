from abc import ABC, abstractmethod

from rtvoice.events.models import VoiceAssistantEvent
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.state.models import StateType
from rtvoice.state.state_machine import (
    VoiceAssistantContext,
    VoiceAssistantStateMachine,
)


class AssistantState(ABC, LoggingMixin):
    def __init__(self):
        self._state_machine: VoiceAssistantStateMachine | None = None

    def set_state_machine(self, state_machine: VoiceAssistantStateMachine) -> None:
        self._state_machine = state_machine

    @property
    @abstractmethod
    def state_type(self) -> StateType:
        pass

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        pass

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        pass

    @abstractmethod
    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext | None = None
    ) -> None:
        pass

    async def _transition_to(self, new_state: "AssistantState") -> None:
        if self._state_machine is None:
            raise RuntimeError("State machine not set - cannot transition")
        await self._state_machine.transition_to(new_state)

    async def _transition_to_idle(self) -> None:
        from rtvoice.state.idle import IdleState

        await self._transition_to(IdleState())

    async def _transition_to_listening(self) -> None:
        from rtvoice.state.listening import ListeningState

        await self._transition_to(ListeningState())

    async def _transition_to_responding(self) -> None:
        from rtvoice.state.responding import RespondingState

        await self._transition_to(RespondingState())

    async def _transition_to_tool_calling(self) -> None:
        from rtvoice.state.tool_calling import ToolCallingState

        await self._transition_to(ToolCallingState())

    async def _transition_to_timeout(self) -> None:
        from rtvoice.state.timeout import TimeoutState

        await self._transition_to(TimeoutState())
