"""Centralized logging configuration using loguru."""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from .config import get_config


def setup_logger() -> None:
    """Configure logger based on configuration settings.

    Sets up file and console logging with appropriate formatting and rotation.
    """
    try:
        config = get_config()
        log_config = config.logging
    except RuntimeError:
        # If config not loaded yet, use defaults
        log_level = "INFO"
        log_file = "logs/ragagent2.log"
        log_format = "json"
        max_size_mb = 100
        backup_count = 5
    else:
        log_level = log_config.level
        log_file = log_config.file
        log_format = log_config.format
        max_size_mb = log_config.max_size_mb
        backup_count = log_config.backup_count

    # Remove default logger
    logger.remove()

    # Console logging with colors
    if log_format == "json":
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    else:
        console_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        )

    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=True,
    )

    # File logging with rotation
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_format == "json":
        logger.add(
            log_file,
            format="{message}",
            level=log_level,
            rotation=f"{max_size_mb} MB",
            retention=backup_count,
            compression="zip",
            serialize=True,  # JSON format
        )
    else:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation=f"{max_size_mb} MB",
            retention=backup_count,
            compression="zip",
        )

    logger.info("Logger configured successfully", level=log_level, format=log_format)


def log_function_call(func_name: str, **kwargs: Any) -> None:
    """Log a function call with its arguments.

    Args:
        func_name: Name of the function being called
        **kwargs: Function arguments to log
    """
    logger.debug(f"Calling {func_name}", arguments=kwargs)


def log_error(error: Exception, context: str = "") -> None:
    """Log an error with context.

    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
    """
    logger.error(
        f"Error occurred: {str(error)}",
        error_type=type(error).__name__,
        context=context,
        exc_info=True,
    )


def log_metric(metric_name: str, value: float, **tags: Any) -> None:
    """Log a metric value.

    Args:
        metric_name: Name of the metric
        value: Metric value
        **tags: Additional tags for the metric
    """
    logger.info(f"Metric: {metric_name}", value=value, **tags)


# Initialize logger on import
setup_logger()
