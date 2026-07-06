"""Base class for all profile samplers"""

from enum import StrEnum, auto
from typing import Optional, TypeVar

import torch
from torch.distributed import broadcast
from torch.nn import Module

from flexrank.layers.base import ODLayer
from flexrank.types import DistributedInfo, get_single_worker_distributed_info
from flexrank.utils.logger import init_logger
from .registry import register_sampler

log = init_logger(__name__)

Profile = list[int]
FrozenProfile = tuple[int, ...]
Profiles = list[Profile]


class DeployMode(StrEnum):
    """Physical deployment mode for applying a profile.

    ``DeployMode.NO``:
        Apply the profile virtually by changing active inner dimensions only.
        Stored parameter tensors are not pruned.
    ``DeployMode.SVD``:
        Physically prune stored parameter tensors to the selected SVD factors.
    ``DeployMode.GAR``:
        Physically prune stored parameter tensors and use GAR storage for layers
        that support it. Layers without GAR support fall back to SVD storage.

    The member names are intentionally short and stable for user-facing
    configuration: "NO", "SVD", and "GAR".
    """

    NO = auto()
    SVD = auto()
    GAR = auto()


@register_sampler
class BaseSampler:
    """Base class for profile samplers over OD layers.

    The sampler owns three responsibilities:
    1. discovering the OD layers in the wrapped model,
    2. storing or generating candidate profiles for those layers,
    3. synchronizing sampled profiles across distributed workers before use.
    """

    _model: Module
    _od_layers: list[ODLayer]
    _samples: Profiles
    _distr: DistributedInfo

    def __init__(
        self,
        model: Module,
        samples: Optional[Profiles] = None,
        distr: Optional[DistributedInfo] = None,
    ):
        """Initialize the sampler.

        Args:
            model: Model containing `ODLayer` modules to be sampled.
            samples: Optional predefined list of profiles.
            distr: Optional distributed info. If None, assumes single-worker.
        """
        self._model = model
        inferred_device = next(model.parameters()).device
        self._distr = distr or get_single_worker_distributed_info(device=inferred_device)
        self.prepare_sampler(samples)

    def _register_od_layers(self) -> None:
        """Cache the OD layers controlled by this sampler."""
        self._od_layers = [
            module for module in self._model.modules() if isinstance(module, ODLayer)
        ]
        assert len(self._od_layers) > 0, "No OD layers found in the model."

    ProfileType = TypeVar("ProfileType", Profile, Profiles)

    def _broadcast_and_sync(self, samples: ProfileType) -> ProfileType:
        """Broadcast an integer (nested) sequence from worker 0 and return it.

        This helper accepts either a 1D `Profile` or a 2D `Profiles` list and
        performs a single broadcast operation using `torch.tensor`. It returns
        the broadcasted Python list. If the local data differs from worker 0's
        data, an error is logged.
        """
        if not self._distr.is_distributed:
            return samples

        synced = torch.tensor(samples, device=self._distr.device, dtype=torch.int64)
        local = synced.clone()
        broadcast(synced, src=0)
        equal = torch.equal(synced, local)
        if not equal:
            log.error("Samples were not synchronized. Synchronizing to worker 0's samples.")
        return synced.cpu().tolist()

    def _create_samples(self, samples: Optional[Profiles] = None) -> None:
        """Populate the sampler's profile pool.

        When no explicit profiles are provided, the default behavior is to keep a
        single full-width profile available.
        """
        if not samples:
            samples = [self._full_sample()]

        self._samples = samples

    def prepare_sampler(self, samples: Optional[Profiles] = None) -> None:
        """Prepare the sampler by registering layers and creating samples.

        Args:
            samples: Optional profiles.
        """
        self._register_od_layers()
        self._create_samples(samples)
        self._samples_generator = self.sampler()

    @property
    def num_od_layers(self) -> int:
        """Return the number of OD layers controlled by the sampler."""
        return len(self._od_layers)

    def _full_sample(self) -> Profile:
        """Return the full profile with max inner dims for all layers."""
        return [l.max_inner_dim for l in self._od_layers]

    def sampler(self):
        """Generator that yields samples. Default is full sample repeatedly."""
        while True:
            yield self._full_sample()

    def _apply_samples(
        self,
        samples: Profile,
        *,
        deploy_mode: DeployMode = DeployMode.NO,
    ):
        """Apply the sampler to the model.

        Args:
            samples: The profile to apply.
            deploy_mode: Whether and how to physically prune after setting.
        """
        for l, tg_inner_dim in zip(self._od_layers, samples, strict=True):
            l.inner_dim = tg_inner_dim

        if deploy_mode is DeployMode.NO:
            return

        gar_fallback_layers = set()
        for l in self._od_layers:
            use_gar = deploy_mode is DeployMode.GAR and l.supports_gar
            if deploy_mode is DeployMode.GAR and not l.supports_gar:
                gar_fallback_layers.add(l)
            l.prune_weights(use_gar=use_gar)

        if gar_fallback_layers:
            fallback_layers_name = [
                name for name, layer in self._model.named_modules() if layer in gar_fallback_layers
            ]
            log.warning(
                "GAR deployment requested, but %d OD layer(s) do not support GAR "
                "and were deployed with SVD storage instead. Layer names: %s",
                len(fallback_layers_name),
                ", ".join(sorted(fallback_layers_name)),
            )

    def __call__(self, *, return_samples: bool = False) -> FrozenProfile:
        """Sample and apply the next profile.

        Args:
            return_samples: If True, it returns the profile without applying it.

        Returns:
            The sampled profile as a tuple. This explicilty marks mutation
            of the returned object as conceptually wrong and prevents
            accidental side-effects.
        """
        samples = next(self._samples_generator)
        samples = self._broadcast_and_sync(samples)
        if not return_samples:
            self._apply_samples(samples)
        return tuple(samples)

    def set_p_for_layers(
        self,
        samples: Profile,
        *,
        deploy_mode: DeployMode = DeployMode.NO,
    ) -> None:
        """Set the profile for layers.

        Args:
            samples: The profile to set.
            deploy_mode: Whether and how to physically prune after setting.
        """
        self._apply_samples(samples, deploy_mode=deploy_mode)

    def reset_to_full(self) -> None:
        """Reset all layers to their maximum inner dimensions."""
        for layer in self._od_layers:
            layer.inner_dim = layer.max_inner_dim

    @property
    def distributed_info(self) -> DistributedInfo:
        """Get the distributed info used by the sampler."""
        return self._distr
