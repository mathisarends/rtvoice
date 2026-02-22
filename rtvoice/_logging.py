import logging
import os

from dotenv import load_dotenv

load_dotenv(override=True)

LIBRARY_NAME = "rtvoice"

logger = logging.getLogger(LIBRARY_NAME)
logger.addHandler(logging.NullHandler())


def configure_logging(level: str = "WARNING") -> None:
    log_level = getattr(logging, level.upper(), logging.WARNING)

    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.setLevel(log_level)
    logger.addHandler(handler)


_env_level = os.getenv("RTVOICE_LOG_LEVEL")
if _env_level:
    configure_logging(_env_level)
