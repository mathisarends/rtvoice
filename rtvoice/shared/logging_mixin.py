import logging
import os
from typing import ClassVar

from dotenv import load_dotenv

load_dotenv(override=True)

LIBRARY_NAME = "rtvoice"

logger = logging.getLogger(LIBRARY_NAME)
logger.addHandler(logging.NullHandler())


def _configure_library_logging(level: str = "WARNING") -> None:
    log_level = getattr(logging, level.upper(), logging.WARNING)

    library_logger = logging.getLogger(LIBRARY_NAME)

    if library_logger.handlers:
        library_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    library_logger.setLevel(log_level)
    library_logger.addHandler(handler)


def _auto_configure_from_environment() -> None:
    env_log_level = os.getenv("RTVOICE_LOG_LEVEL")

    if env_log_level:
        _configure_library_logging(env_log_level)


_auto_configure_from_environment()


class LoggingMixin:
    logger: ClassVar[logging.Logger] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.logger = logging.getLogger(f"{LIBRARY_NAME}.{cls.__name__}")

    @property
    def instance_logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(
                f"{LIBRARY_NAME}.{self.__class__.__name__}"
            )
        return self._logger
