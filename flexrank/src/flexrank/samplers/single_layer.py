"""Samplers that vary one OD layer at a time."""

from typing import Optional

import numpy as np

from flexrank.utils.logger import init_logger
from .base_sampler import BaseSampler, Profiles
from .registry import register_sampler

log = init_logger(__name__)


class SingleLayerSampler(BaseSampler):
    """Base class for samplers that vary a single OD layer at once."""

    def _get_inner_dims_upto_width(self, width: int):
        """Return candidate inner dimensions for one layer width."""
        raise NotImplementedError

    def _create_samples(self, samples: Optional[Profiles] = None) -> None:
        """Build profiles that modify one OD layer at a time."""
        if samples:
            log.warning(
                "%s received an unexpected argument 'samples'. Ignoring",
                self.__class__.__name__,
            )

        samples_per_layer = []
        for i, l in enumerate(self._od_layers):
            inner_dims = self._get_inner_dims_upto_width(l.max_inner_dim)
            for d in inner_dims:
                sample = [tl.max_inner_dim for tl in self._od_layers]
                sample[i] = d
                samples_per_layer.append(sample)
        self._samples = samples_per_layer

    def sampler(self):
        """Yield one-layer-at-a-time profiles in random order."""
        while True:
            random_perm = np.random.permutation(len(self._samples))
            for idx in random_perm:
                sample = self._samples[idx]
                yield sample


@register_sampler
class SingleLayerLinearSampler(SingleLayerSampler):
    """Sample one layer at a time using linearly spaced widths."""

    def __init__(self, min_p: float, num_models: int, *args, **kwargs) -> None:
        """Initialize the linear single-layer sampler."""
        if not min_p:
            min_p = 1.0 / num_models
        self.min_p = min_p
        self.num_models = num_models
        super().__init__(*args, **kwargs)

    def _get_inner_dims_upto_width(self, width: int) -> list[int]:
        """Return linearly spaced inner dimensions between `min_p` and `width`."""
        min_inner_dim = int(np.ceil(width * self.min_p))
        return np.linspace(min_inner_dim, width, self.num_models, dtype=int).tolist()


@register_sampler
class SingleLayerPowerTwoSampler(SingleLayerSampler):
    """Sample one layer at a time over powers-of-two widths."""

    def _get_inner_dims_upto_width(self, width: int) -> list[int]:
        """Return powers-of-two inner dimensions up to `width`."""
        return [2**i for i in range(1, int(np.ceil(np.log2(width))))] + [width]
