"""Logging configuration for experiments."""
import logging
import sys


def get_logger(name: str = "ml-flood-ungauged", level: int = logging.INFO) -> logging.Logger:
    """
    Get a standardized logger with stream formatting.
    
    Args:
        name: Name of the logger module.
        level: Logging verbosity level.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
