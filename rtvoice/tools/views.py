from typing import Any

from pydantic import BaseModel, ConfigDict

from rtvoice.events.bus import EventBus


class SpecialToolParameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    event_bus: EventBus | None = None
    context: Any | None = None
