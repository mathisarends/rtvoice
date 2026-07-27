import logging
from typing import Any, Self

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

logger = logging.getLogger(__name__)


class SpeechSpeed(float):
    """Clamps to the range the Realtime API accepts, so no caller can put an
    out-of-range speed on the wire."""

    _MIN = 0.25
    _MAX = 1.5

    def __new__(cls, value: float) -> Self:
        clamped = max(cls._MIN, min(float(value), cls._MAX))

        if value != clamped:
            logger.warning(
                "Speech speed %.2f is out of range [%.2f, %.2f], clipping to %.2f",
                value,
                cls._MIN,
                cls._MAX,
                clamped,
            )

        return super().__new__(cls, clamped)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls, core_schema.float_schema()
        )


DEFAULT_SPEECH_SPEED = SpeechSpeed(1.0)
