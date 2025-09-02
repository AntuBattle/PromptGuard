import logging
import sys
from typing import Any, Literal, TextIO

import structlog

LOGGER = structlog.get_logger(__name__)

LOG_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
EXTERNAL_LOGGERS = {
    "transformers",
    "presidio-analyzer",
}


def configure_logger(
    log_level: LOG_LEVELS = "INFO",
    render_json: bool = False,
    stream: TextIO = sys.stdout,
):
    """
    Configures the logger for the package.

    Args:
        log_level: The log level to use for the logger. It should be one of the following strings:
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        render_json: Whether to render log messages in JSON format. Default is False.
        stream: The stream to write log messages to. Default is sys.stdout.
    """
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=stream,
    )

    log_level_to_int = {
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.FATAL,
        "ERROR": logging.ERROR,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    render_processors = [structlog.dev.ConsoleRenderer()]
    if render_json:
        render_processors = [structlog.processors.JSONRenderer()]

    structlog.configure(
        context_class=dict,
        wrapper_class=structlog.make_filtering_bound_logger(log_level_to_int[log_level]),
        logger_factory=structlog.PrintLoggerFactory(stream),
        cache_logger_on_first_use=False,
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.dict_tracebacks,
        ]
        + render_processors,
    )
    for log_name in EXTERNAL_LOGGERS:
        logging.getLogger(log_name).setLevel(logging.WARNING)


def _get_library_name() -> str:
    return __name__.split(".")[0]


def get_logger(name: str | None = None) -> Any:
    """
    Returns the logger for the package.

    Returns:
        The logger for the package.
    """

    if name is None:
        name = _get_library_name()

    return structlog.get_logger(name)
