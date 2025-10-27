import logging
import logging.handlers
import sys
from pathlib import Path


# Log level mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Custom log format
LOG_FORMAT = "[%(asctime)s | %(levelname)s | %(custom_word)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class CustomFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt, custom_word):
        super().__init__(fmt, datefmt)
        self.custom_word = custom_word

    def format(self, record):
        # Rename INFO to INFO1 (you can extend this with a dict if needed)
        if record.levelno == logging.INFO:
            record.levelname = "  INFO "
        elif record.levelno == logging.WARNING:
            record.levelname = "WARNING"
        elif record.levelno == logging.ERROR:
            record.levelname = " ERROR "
        # Add the custom word to the record
        record.custom_word = self.custom_word
        return super().format(record)


# Colored logging for console output
class ColoredFormatter(CustomFormatter):
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[0m",
        "WARNING": "\033[93m",  # Yellow
        " ERROR ": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record):
        log_msg = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_msg}{self.COLORS['RESET']}"


CUSTOM_WORD = None

def get_logger(name="petharbor", level="debug", log_dir=None, method=None):
    """Configures and returns a logger with both console and file handlers."""
    if log_dir:
        try:
            LOG_DIR = Path(log_dir)
            LOG_DIR.mkdir(exist_ok=True)
            LOG_FILE = f"{LOG_DIR}/{name}.log"
        except Exception as e:
            raise Exception(f"Failed to create log directory: {e}")

    if method:
        global CUSTOM_WORD
        CUSTOM_WORD = f"PetHarbor-{method}"

    level = LOG_LEVELS.get(level.lower(), logging.DEBUG)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(LOG_FORMAT, DATE_FORMAT, CUSTOM_WORD))
    logger.addHandler(console_handler)

    if log_dir:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(CustomFormatter(LOG_FORMAT, DATE_FORMAT, CUSTOM_WORD))
        logger.addHandler(file_handler)

    # Prevent logs from being passed to the root logger
    logger.propagate = False

    return logger


# Usage example
if __name__ == "__main__":
    log = get_logger("test_logger", "debug")

    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
    log.critical("This is a critical message.")
