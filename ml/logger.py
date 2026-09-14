import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Configures and returns a standard logger for ML package modules.

    Args:
        name: Name of the logger, typically __name__.
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    return logger
