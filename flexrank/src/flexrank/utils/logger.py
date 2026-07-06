"""Logging helpers with optional main-process-only emission policy."""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, Optional

from flexrank.types import DistributedInfo

__all__ = [
    "init_logger",
    "set_distributed_logging_state",
    "format_elapsed_time",
    "timed",
]


class _LoggingState:  # pylint: disable=too-few-public-methods
    """Mutable module-local state used by logging filters."""

    distr: DistributedInfo | None = None


_STATE = _LoggingState()


class MainProcessFilter(logging.Filter):  # pylint: disable=too-few-public-methods
    """Emit records only from the main process when distributed state is set."""

    def filter(self, record: logging.LogRecord) -> bool:
        distr = _STATE.distr
        if distr is None:
            return True
        return distr.is_main_process


def set_distributed_logging_state(distr: DistributedInfo | None) -> None:
    """Set the distributed state consulted by logger filters."""
    _STATE.distr = distr


def init_logger(module_name: str) -> logging.Logger:
    """Create or return a module logger with main-process filtering."""
    logger = logging.getLogger(module_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s (%(name)s:%(lineno)d): %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handler.addFilter(MainProcessFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def format_elapsed_time(seconds):
    """Format a duration in seconds into a human-readable unit."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} nanoseconds"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} microseconds"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} milliseconds"
    if seconds < 60:
        return f"{seconds:.2f} seconds"

    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)} minutes, {remaining_seconds:.2f} seconds"


def timed(
    name: Optional[str] = None, logger: Optional[logging.Logger] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a function and report its execution time."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper_timer(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            formatted_time = format_elapsed_time(elapsed_time)

            print_name = name or f"Function '{func.__name__}'"
            print_msg = f"{print_name} took: {formatted_time}"

            if logger:
                logger.info(print_msg)
            else:
                print(print_msg)

            return result

        return wrapper_timer

    return decorator
