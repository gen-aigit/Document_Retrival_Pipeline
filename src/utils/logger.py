"""Shared logging setup. Every module gets a logger that writes to both the
console and the API log file configured in .env."""

import logging

from src.config import settings

_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(settings.log_level)
    logger.propagate = False

    settings.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(settings.log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger
