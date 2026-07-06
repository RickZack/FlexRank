"""Distributed runtime type adapters."""

from enum import StrEnum, auto
from typing import NamedTuple, Optional

import torch


class DistributedType(StrEnum):
    """Compatibility bridge with `accelerate.state.DistributedType`"""

    @staticmethod
    def _generate_next_value_(name, *args):  # pylint: disable=unused-argument
        return name.upper()

    NO = auto()
    MULTI_CPU = auto()
    MULTI_GPU = auto()
    MULTI_NPU = auto()
    MULTI_MLU = auto()
    MULTI_SDAA = auto()
    MULTI_MUSA = auto()
    MULTI_XPU = auto()
    DEEPSPEED = auto()
    FSDP = auto()
    XLA = auto()
    MEGATRON_LM = auto()
    MULTI_HPU = auto()
    MULTI_NEURON = auto()


class DistributedInfo(NamedTuple):
    """Information about the distributed environment. Compatibility bridge with
    `accelerate.state.PartialState`"""

    process_index: int
    is_main_process: bool
    is_local_main_process: bool
    num_processes: int
    device: torch.device
    mixed_precision: str
    distributed_type: DistributedType

    @property
    def is_distributed(self) -> bool:
        """Whether the distributed type is not NO."""
        return self.distributed_type != DistributedType.NO

    @property
    def is_fsdp(self) -> bool:
        """Whether the distributed type is FSDP."""
        return self.distributed_type == DistributedType.FSDP


def get_single_worker_distributed_info(
    device: Optional[torch.device] = None, mixed_precision: str = "no"
) -> DistributedInfo:
    """Create a DistributedInfo for a single-worker setup."""
    if device is None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    return DistributedInfo(
        process_index=0,
        is_main_process=True,
        is_local_main_process=True,
        num_processes=1,
        device=device,
        mixed_precision=mixed_precision,
        distributed_type=DistributedType.NO,
    )
