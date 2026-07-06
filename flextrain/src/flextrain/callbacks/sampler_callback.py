"""Sampler callback integration for Hugging Face Trainer."""

import weakref
from typing import TypeAlias, Union

import numpy as np
from transformers import Trainer, TrainerCallback

from flexrank.samplers.base_sampler import BaseSampler
from flexrank.utils import init_logger
from flextrain.utils import get_distributed_info

__all__ = ["SamplerCallback"]

TrainerProxy: TypeAlias = Union[Trainer, weakref.ProxyType]

logger = init_logger(__name__)


class SamplerCallback(TrainerCallback):
    """Apply sampler decisions around each training step.

    The callback stays intentionally thin: the sampler decides how to draw and
    synchronize profiles, while the callback only wires those decisions into the
    Trainer lifecycle.
    """

    def __init__(self, trainer: Trainer, profiles: list[np.ndarray]):
        """Store the trainer reference and the profiles used to build a sampler."""
        self.trainer: TrainerProxy = weakref.proxy(trainer)
        self.sampler = None
        self.profiles = profiles

    def init_sampler(self, sampler_cls: type[BaseSampler], sampler_kwargs: dict):
        """Instantiate the configured sampler for the trainer's model."""
        sampler_kwargs = sampler_kwargs.copy()
        if "model" in sampler_kwargs:
            logger.warning(
                "'model' keyword argument shall not be passed as sampler keyword "
                "arguments. Overriden by Trainer model"
            )
        sampler_kwargs["model"] = self.trainer.model
        sampler_kwargs["distr"] = get_distributed_info(self.trainer.accelerator)
        self.sampler = sampler_cls(**sampler_kwargs)

    def on_step_begin(self, _args, _state, _control, **kwargs):
        """Sample and apply the next synchronized profile before each train step."""
        model = kwargs["model"]
        if model.training:
            self.sampler()

    def on_step_end(self, _args, _state, _control, **kwargs):
        """Restore the full-width model after each optimization step."""
        self.sampler.reset_to_full()

    def on_train_end(self, _args, _state, _control, **kwargs):
        """Leave the model in its full-width state once training is over."""
        self.sampler.reset_to_full()
        self.trainer.pop_callback(self)
