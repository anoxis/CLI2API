"""Structured logging configuration."""

import logging
import sys
from datetime import datetime
from typing import Any

import json


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging.
    
    Outputs log records as JSON objects with consistent structure:
    - timestamp: ISO 8601 format
    - level: Log level (INFO, ERROR, etc.)
    - logger: Logger name
    - message: Log message
    - extra: Additional context fields
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format.
            
        Returns:
            JSON-formatted log string.
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields (avoid built-in attributes)
        skip_keys = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "message", "pathname", "process", "processName", "relativeCreated",
            "thread", "threadName", "exc_info", "exc_text", "stack_info",
        }
        
        for key, value in record.__dict__.items():
            if key not in skip_keys and not key.startswith("_"):
                log_data[key] = value

        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    logger_name: str | None = None,
) -> logging.Logger:
    """Configure structured logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Use JSON formatter if True, else use simple text format.
        logger_name: Logger name (default: root logger).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level.upper())
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level.upper())
    
    # Set formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    if logger_name:
        logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name.
    
    Args:
        name: Logger name (typically __name__ of the module).
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
