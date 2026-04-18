"""Centralized logging setup for Ollama TUI."""

import logging
import os
from pathlib import Path


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Configure and return a named logger with file and optional console output.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default: DEBUG for file, WARNING for console).

    Returns:
        Configured Logger instance.
    """
    log_dir = Path.home() / ".ollama_tui" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ollama_tui.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers in interactive sessions
    if logger.handlers:
        return logger

    # File handler — always write DEBUG+ to log file
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
