from pydantic import BaseModel

from rtvoice.events import EventBus


class SpecialToolParameters(BaseModel):
    event_bus: EventBus
