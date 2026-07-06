"""FlexRank trainers built on top of the standard CV trainer.

These trainers target single-worker CV fine-tuning/compression workflows and
are mainly used inside notebooks for experimentation.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor
from flexrank.samplers.base_sampler import BaseSampler
from flexrank.trainers.standard_trainer import (
    StandardTrainer,
    TrainingArguments,
)
from flexrank.utils.logger import init_logger

__all__ = ["FlexRankTrainer", "FlexRankTrainArguments"]

logger = init_logger(__name__)


@dataclass
class FlexRankTrainArguments(TrainingArguments):
    """Training configuration used by `FlexRankTrainer`."""

    sampler_cls: Optional[type[BaseSampler]] = field(default=None)
    sampler_kwargs: dict = field(default_factory=dict)


@dataclass
class FlexRankTrainer(StandardTrainer):
    """Single-worker CV trainer with profile sampling/distillation support."""

    args: FlexRankTrainArguments = field(default_factory=FlexRankTrainArguments)
    sampler: BaseSampler = field(init=False)
    teacher: torch.nn.Module | None = field(init=False, default=None)
    do_distillation: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.do_distillation:
            self.teacher = deepcopy(self.model).requires_grad_(False).eval()

        self._init_sampler(self.args.sampler_cls, self.args.sampler_kwargs)

    def _init_sampler(self, sampler_cls: type[BaseSampler], sampler_kwargs: dict):
        sampler_kwargs = sampler_kwargs.copy()
        if "model" in sampler_kwargs:
            logger.warning(
                "'model' keyword argument shall not be passed as sampler keyword "
                "arguments. Overriden by Trainer model"
            )
        sampler_kwargs["model"] = self.model
        self.sampler = sampler_cls(**sampler_kwargs)

    def _forward_backward(
        self, data: Tensor, target: Tensor | tuple[Tensor, Tensor], criterion: torch.nn.Module
    ) -> tuple[Tensor, Tensor]:
        self.sampler()
        if isinstance(target, tuple):
            _labels, soft_target = target
            return super()._forward_backward(data, soft_target, criterion)
        return super()._forward_backward(data, target, criterion)

    def _train_dataloader_iterator(self):

        if not self.teacher:
            return super()._train_dataloader_iterator()

        def get_teacher_out(inputs: Tensor):
            with torch.no_grad():
                return self.teacher(inputs).softmax(dim=1)

        train_loader = self._get_train_dataloader()
        device = self.args.distr.device

        return (
            (
                d.to(device),
                (t.to(device), get_teacher_out(d.to(device))),
            )
            for d, t in train_loader
        )

    def _on_train_epoch_end(self) -> None:
        self.sampler.reset_to_full()

    def _get_target_labels(self, target: Tensor | tuple[Tensor, Tensor]) -> Tensor:
        if isinstance(target, tuple):
            labels, _soft_target = target
            return labels
        if target.ndim == 1:
            return target
        if target.ndim == 2:
            return target.argmax(dim=1)
        raise ValueError(f"Unexpected target shape for FlexRankTrainer: {target.shape}")

    @property
    def num_od_layers(self) -> int:
        """The number of `ODLayer`s in the model"""
        return self.sampler.num_od_layers

    @property
    def uses_distillation(self) -> bool:
        """Whether the Trainer is using teacher's soft targets or hard labels"""
        return bool(self.teacher)
