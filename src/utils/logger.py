import logging
import sys

from src.utils.singleton import SingletonMeta

RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[36m",    # Cyan
    "INFO": "\033[32m",     # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",    # Red
    "CRITICAL": "\033[41m\033[37m", # Red Background, White Text
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLORS.get(record.levelname, RESET)
        record.levelname = f"{log_color}{record.levelname:<8}{RESET}"
        return super().format(record)

class Logger(metaclass=SingletonMeta):
    """Utility class that handles logging with colors"""

    def __init__(self):
        self._logger = logging.getLogger("app_logger")
        self._logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter(
            fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self._logger.addHandler(handler)

    # Route methods safely using the instance initialization helper
    def debug(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self._logger.critical(msg, *args, **kwargs)