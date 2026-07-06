"""Distributed runtime and utility helpers for FlexTrain."""

from functools import wraps
import contextlib
import logging
import os

import torch
from accelerate import Accelerator, PartialState
from flexrank.types import DistributedInfo, DistributedType
from flexrank.utils import init_logger as _init_logger
from flexrank.utils import set_distributed_logging_state

log = _init_logger(__name__)


def suppress_stdout(func):
    """Decorator to suppress stdout of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            return func(*args, **kwargs)

    return wrapper


def make_model_contiguous(model: torch.nn.Module):
    """Make the model parameters contiguous in memory."""
    for p in model.parameters():
        p.data = p.data.contiguous()


def get_distributed_info(src: Accelerator | PartialState) -> DistributedInfo:
    """Get distributed info from the accelerator."""
    return DistributedInfo(
        process_index=src.process_index,
        is_main_process=src.is_main_process,
        is_local_main_process=src.is_local_main_process,
        num_processes=src.num_processes,
        device=src.device,
        mixed_precision=getattr(src, "mixed_precision", "no"),
        distributed_type=DistributedType(src.distributed_type.value),
    )


def init_logging_state():
    """Initialize logging state for distributed training."""
    dist_info = get_distributed_info(PartialState())
    set_distributed_logging_state(dist_info)


def init_logger(module_name: str) -> logging.Logger:
    """Initialize distributed logging state and create a module logger."""
    init_logging_state()
    return _init_logger(module_name)
