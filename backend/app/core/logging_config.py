import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


APP_LOGGER_NAME="app"

LOG_DIR=Path(
    os.getenv(
        "LOG_DIR",
        Path.home() / ".local" / "state" / "ai-streaming" / "logs" ,
    )
)

LOG_FILE=LOG_DIR / "app.log"

LOG_MAX_BYTES=10*1024*1024
LOG_BACKUP_COUNT=5

LOG_FORMAT=(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

def configure_logging() -> None:

    logger = logging.getLogger(APP_LOGGER_NAME)

    logger.setLevel(logging.INFO)

    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    if logger.handlers:
        return

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)


    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_handler=RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
