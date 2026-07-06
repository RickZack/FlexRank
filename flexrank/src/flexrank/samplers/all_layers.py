"""Samplers that vary all OD layers jointly."""

from typing import Optional

import numpy as np

from flexrank.utils.logger import init_logger
from .base_sampler import BaseSampler, Profiles
from .registry import register_sampler

log = init_logger(__name__)


class AllLayerSampler(BaseSampler):
    """Base class for samplers that scale all OD layers together."""

    def _get_ps_for_all_layers(self) -> np.ndarray:
        """Return the sampling ratios to apply to each layer width."""
        raise NotImplementedError

    def _create_samples(self, samples: Optional[Profiles] = None) -> None:
        """Build layer profiles by applying each ratio to every OD layer."""
        if samples:
            log.warning(
                "%s received an unexpected argument 'samples'. Ignoring",
                self.__class__.__name__,
            )

        samples_for_all_layers = []
        for p in self._get_ps_for_all_layers():
            sample = [int(np.ceil(p * l.max_inner_dim)) for l in self._od_layers]
            samples_for_all_layers.append(sample)
        self._samples = samples_for_all_layers


@register_sampler
class AllLayerLinearSampler(AllLayerSampler):
    """Sample all layers using a shared linearly spaced width ratio."""

    def __init__(
        self,
        min_p: float,
        num_models: int,
        *args,
        sandwich: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the linear all-layer sampler."""
        if not min_p:
            min_p = 1.0 / num_models
        self.min_p = min_p
        self.num_models = num_models
        super().__init__(*args, **kwargs)
        self.sandwich = sandwich
        self.min_sample = [int(np.ceil(min_p * l.max_inner_dim)) for l in self._od_layers]
        self.max_sample = [l.max_inner_dim for l in self._od_layers]

    def _get_ps_for_all_layers(self) -> np.ndarray:
        """Return ratios linearly spaced between `1.0` and `min_p`."""
        return np.linspace(self.min_p, 1.0, self.num_models)[::-1]

    def sampler(self):
        """Yield sampled profiles, optionally in sandwich order."""
        rng = np.random.default_rng()
        while True:
            random_perm = rng.permutation(len(self._samples))
            for idx in random_perm:
                sample = self._samples[idx]
                if self.sandwich:
                    yield self.max_sample
                    yield sample
                    yield self.min_sample
                else:
                    yield sample


@register_sampler
class AllLayerIndependentRandomSampler(BaseSampler):
    """Sample each layer independently from a shared width grid."""

    def __init__(self, min_p: float, num_models: int, *args, **kwargs) -> None:
        """Initialize the independent all-layer sampler."""
        if not min_p:
            min_p = 1.0 / num_models
        self.min_p = min_p
        self.num_models = num_models
        super().__init__(*args, **kwargs)

    def _create_samples(self, samples: Optional[Profiles] = None) -> None:
        """Ignore explicit samples and keep the default full-width sample pool."""
        if samples:
            log.warning(
                "%s received an unexpected argument 'samples'. Ignoring",
                self.__class__.__name__,
            )
        super()._create_samples()

    def _get_inner_dims_upto_width(self, width: int) -> np.ndarray:
        """Return the candidate inner dimensions for a single layer width."""
        min_inner_dim = int(np.ceil(self.min_p * width))
        return np.linspace(min_inner_dim, width, self.num_models, dtype=int)

    def sampler(self):
        """Yield profiles by sampling each layer independently."""
        rng = np.random.default_rng()
        options_per_layer = [
            self._get_inner_dims_upto_width(layer.max_inner_dim) for layer in self._od_layers
        ]
        while True:
            yield [rng.choice(options) for options in options_per_layer]
