from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict

from rtvoice.conversation import ConversationHistory
from rtvoice.events.bus import EventBus


class _Inject:
    """Marker for parameters that should be injected from ToolContext."""


_INJECT_MARKER = _Inject()


if TYPE_CHECKING:
    type Inject[T] = T
else:

    class Inject:
        """Marks a parameter for dependency injection from ToolContext.

        Usage: ``event_bus: Inject[EventBus]``
        """

        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[item, _INJECT_MARKER]


class ToolContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    event_bus: EventBus | None = None
    context: Any | None = None
    conversation_history: ConversationHistory | None = None
