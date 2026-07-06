"""Utilities for profiling the layers, useful for profile search algorithms."""

import bisect
from dataclasses import dataclass, field
from typing import Any, Optional, Union, TypeAlias

import numpy as np
import torch
from tqdm import tqdm

from flexrank.types import DistributedInfo, get_single_worker_distributed_info
from flexrank.utils.logger import init_logger

from ..layers.base import ODLayer
from ..samplers.base_sampler import BaseSampler
from ..trainers.base_trainer import BaseEvaluator

logger = init_logger(__name__)

__all__ = [
    "LayerStats",
    "ProfilesData",
    "MergedProfilesData",
    "profile_size_only",
    "profile_model",
    "fast_profile_model",
]

DatasetType: TypeAlias = Union[torch.utils.data.Dataset, torch.utils.data.DataLoader]


@dataclass(frozen=True)
class LayerStats:
    """
    Stores evaluation metrics for rank-compression of a single layer .

    Attributes:
        inner_dims: A list of integers representing the evaluated ranks.
        errors: A list of floats representing the loss at respective rank.
        params_savings: A list of integers indicating the number of parameters saved
                        for each evaluated rank.
    """

    inner_dims: list[int]
    errors: list[float]
    params_savings: list[int]


@dataclass(frozen=True)
class BaseProfilesData:
    """
    Stores profiles data

    Attributes:
        layers_name: A list of strings, where each string is the name
                     of a layer included in the profile.
        profiles: A list of lists, each holding the p-values for the layers
                  included in the profile.
        params: A list of integers representing the number of parameters for each profile
    """

    layers_name: list[str]
    profiles: list[list[float]]
    params: list[int]

    def evaluate_profiles(
        self,
        eval_trainer: BaseEvaluator,
        eval_ds: Optional[DatasetType] = None,
        *,
        distr: Optional[DistributedInfo] = None,
        metric_key_prefix: str = "eval",
        verbose: bool = False,
    ) -> "ProfilesData":
        """
        Evaluate a sequence of model profiles and collect their evaluation metrics.

        Parameters
        ----------
        eval_trainer : BaseEvaluator used to evaluate the model.
        eval_ds : DatasetType, optional
           The dataset to use for evaluation. If None, defaults to the
           trainer's evaluation dataset (default: None).
        distr : DistributedInfo, optional
            Object summarizing distributed status (default: None).
        metric_key_prefix : str, optional
            A prefix to add to the keys of the returned metrics dictionary (default: "eval").
        verbose : bool, optional
            If True, log progress for each profile using the module logger (default: False).

        Returns
        -------
        ProfilesData
            A new ProfilesData constructed from:
            - the same `layers_name` as `prof_stats`,
            - the same `profiles` sequence,
            - `metrics` : a list of metric mappings (one per evaluated profile),
            - the original `params` sequence from `prof_stats`.
        """
        prediction_loss_only = True if eval_trainer.compute_metrics is None else None
        distr = distr or get_single_worker_distributed_info()
        sampler = BaseSampler(eval_trainer.model, distr=distr)
        metrics = []
        msg_part = ["Submodel Evaluation"]

        pbar = tqdm(
            zip(self.profiles, self.params),
            total=len(self.profiles),
            desc="Evaluating submodels",
            disable=not distr.is_main_process,
        )

        for i, (profile, params) in enumerate(pbar, start=1):
            sampler.set_p_for_layers(profile)
            pbar.set_postfix({"submodel": f"{i}/{len(self.profiles)}", "params": params})

            metric = eval_trainer.evaluation_loop(
                description="Submodel Evaluation",
                dataloader=eval_trainer.get_eval_dataloader(eval_ds),
                metric_key_prefix=metric_key_prefix,
                prediction_loss_only=prediction_loss_only,
            ).metrics
            metrics.append(metric)

            if verbose:
                metric_display = ",".join(
                    f"{k}: {v:.4f}"
                    for k, v in metric.items()
                    if any(m in k for m in ("loss", "accuracy"))
                )
                msg_part.append(
                    f"Profile {i}/{len(self.profiles)}: Params={params}, {metric_display}"
                )
        if verbose:
            logger.info("\n".join(msg_part))

        sampler.reset_to_full()

        return ProfilesData(
            layers_name=self.layers_name,
            profiles=self.profiles,
            params=self.params,
            metrics=metrics,
        )

    def get_profile_for_params(self, param_threshold: int) -> tuple[list[float], int]:
        """
        Get the profile that meets the parameter threshold.

        Profiles are assumed to be sorted by decreasing params (largest first).
        Returns the largest profile with params <= param_threshold.

        Args:
            param_threshold: Maximum number of parameters allowed.

        Returns:
            Tuple of (profile, params, index) for the selected profile.

        Raises:
            ValueError: If no profile meets the threshold.
        """
        # self.params is descending: [1000, 800, 600, 400]
        # For binary search on descending list, use negation to make it ascending
        # -params: [-1000, -800, -600, -400] (ascending)
        # bisect_left finds first index where -params[i] >= -threshold
        # i.e., first index where params[i] <= threshold
        neg_params = [-p for p in self.params]
        idx = bisect.bisect_left(neg_params, -param_threshold)

        if idx >= len(self.params):
            raise ValueError(
                f"No profile found with params <= {param_threshold}. "
                f"Smallest available: {self.params[-1]}"
            )

        return self.profiles[idx], self.params[idx]

    @property
    def max_params(self) -> int:
        """Return the maximum number of parameters across all profiles."""
        return self.params[0]

    @property
    def min_params(self) -> int:
        """Return the minimum number of parameters across all profiles."""
        return self.params[-1]

    def to_export_dict(self) -> dict[str, Any]:
        """Return deployment-critical profile data without evaluation metrics."""
        return {
            "layers_name": self.layers_name,
            "profiles": self.profiles,
            "params": self.params,
        }


@dataclass(frozen=True)
class ProfilesData(BaseProfilesData):
    """
    Stores profiles data and associated metrics

    Attributes:
        layers_name: A list of strings, where each string is the name
                     of a layer included in the profile.
        profiles: A list of lists, each holding the p-values for the layers
                  included in the profile.
        params: A list of integers representing the number of parameters for each profile
        metrics: A list of dictionaries, one for each profile, containing the evaluation metrics
                 calculated from the profiles. Keys are metric names (str), values are numerical.
    """

    metrics: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class MergedProfilesData(BaseProfilesData):
    """
    Stores profiles data and associated predicted and evaluated metrics for profiles.

    Attributes:
        layers_name: A list of strings, where each string is the name
                     of a layer included in the profile.
        profiles: A list of lists, each holding the p-values for the layers
                  included in the profile.
        params: A list of integers representing the number of parameters for each profile
        pred_metrics: A list of dictionaries, one for each profile, containing the **predicted**
                      metrics from the profiles. Keys are metric names (str), values are numerical.
        eval_metrics: A list of dictionaries, one for each profile, containing the **evaluated**
                      metrics from the profiles.Keys are metric names (str), values are numerical.
    """

    pred_metrics: list[dict[str, float]]
    calib_metrics: list[dict[str, float]]
    eval_metrics: list[dict[str, float]]

    @staticmethod
    def from_profiles_data(
        pred_profiles_data: ProfilesData,
        calib_profiles_data: ProfilesData,
        eval_profiles_data: ProfilesData,
    ) -> "MergedProfilesData":
        """
        Create a MergedProfilesData instance from predicted and evaluated ProfilesData objects.

        Merges two ProfilesData objects (one with predicted metrics and one with evaluated metrics)
        into a single MergedProfilesData object. Both input ProfilesData objects must have matching
        layer names and profiles.

        Args:
            pred_profiles_data (ProfilesData): object containing predicted metrics.
            calib_profiles_data (ProfilesData): object containing eval metrics on calibration set.
            eval_profiles_data (ProfilesData): object containing eval metrics.

        Returns:
            MergedProfilesData: A new MergedProfilesData instance combining both predicted and
                evaluated metrics along with shared layers, profiles, and parameters.

        Raises:
            AssertionError: If layer names do not match between pred_* and eval_*.
            AssertionError: If profiles do not match between pred_*, calib_* and eval_*.
        """
        assert pred_profiles_data.layers_name == eval_profiles_data.layers_name, (
            "Layer names must match between predicted and evaluated profiles data"
        )
        assert (
            pred_profiles_data.profiles == calib_profiles_data.profiles
            and pred_profiles_data.profiles == eval_profiles_data.profiles
        ), "Profiles must match between predicted and evaluated profiles data"

        return MergedProfilesData(
            layers_name=pred_profiles_data.layers_name,
            profiles=pred_profiles_data.profiles,
            pred_metrics=pred_profiles_data.metrics,
            calib_metrics=calib_profiles_data.metrics,
            eval_metrics=eval_profiles_data.metrics,
            params=pred_profiles_data.params,
        )


def _check_lin_conv(layer):
    return isinstance(layer, ODLayer)


def _p_to_inner_dim(p: float, inner_dim: int) -> int:
    return int(np.ceil(inner_dim * p))


def _sample_inner_dims(layer: ODLayer, min_p: float, n_cut_points: int) -> list[int]:
    min_inner_dim = _p_to_inner_dim(min_p, layer.max_inner_dim)
    num_inner_dims = min(layer.max_inner_dim - min_inner_dim + 1, n_cut_points)
    inner_dims = np.linspace(min_inner_dim, layer.max_inner_dim, num_inner_dims, dtype=int)
    if min_inner_dim != 1:
        inner_dims = np.insert(inner_dims, 0, 1)
    return inner_dims.tolist()


def _sample_uniform_ps(min_p: float, n_cut_points: int) -> list[float]:
    return np.linspace(min_p, 1.0, n_cut_points).tolist()


def _sample_inner_dims_from_p_grid(
    layer: ODLayer,
    min_p: float,
    n_cut_points: int,
) -> list[int]:
    sampled_ps = _sample_uniform_ps(min_p, n_cut_points)
    return [_p_to_inner_dim(p, layer.max_inner_dim) for p in sampled_ps]


def _record_param_saving(layer: ODLayer, inner_dims: list[int]) -> list[int]:
    p_savings = []
    base_params = layer.get_num_parameters(layer.max_inner_dim)
    for current_inner_dim in inner_dims:
        num_params = layer.get_num_parameters(current_inner_dim)
        p_savings.append(base_params - num_params)

    return p_savings


def _profile_layer(
    layer: ODLayer, evaluator: BaseEvaluator, inner_dims: list[int], full_loss: float
) -> LayerStats:
    errors = []
    params_savings = []
    full_inner_dim = layer.max_inner_dim
    base_params = layer.get_num_parameters(full_inner_dim)
    pbar = tqdm(inner_dims, leave=False, disable=not evaluator.distr.is_local_main_process)

    for current_inner_dim in pbar:
        layer.inner_dim = current_inner_dim
        current_loss = evaluator.evaluate()["eval_loss"]
        errors.append(current_loss - full_loss)
        pbar.set_description(
            f"Rank: {current_inner_dim}/{full_inner_dim} - Loss: {current_loss:.4f}/{full_loss:.4f}"
        )
        params_savings.append(base_params - layer.get_num_parameters(current_inner_dim))

    layer.inner_dim = full_inner_dim  # reset to full rank after evaluation

    return LayerStats(inner_dims, errors, params_savings)


def profile_size_only(
    model: torch.nn.Module,
    n_cut_points: int,
    min_p: float,
    *,
    sample_p_grid: bool = False,
) -> dict[str, LayerStats]:
    """Profile model layers by size/parameters only, without computing quantization errors.

    Args:
        model: PyTorch module to profile.
        n_cut_points: Number of cut points for sampling inner dimensions.
        min_p: Minimum proportion threshold for layer profiling.
        sample_p_grid: If True, sample inner dimensions from a predefined p-grid;
            otherwise use uniform sampling. Defaults to False.

    Returns:
        Dictionary mapping layer names to their LayerStats containing inner dimensions
        and parameter savings, with empty error lists.
    """
    layers_stats = {}
    layers = [(n, l) for n, l in model.named_modules() if _check_lin_conv(l)]
    for name, layer in layers:
        if sample_p_grid:
            inner_dims = _sample_inner_dims_from_p_grid(layer, min_p, n_cut_points)
        else:
            inner_dims = _sample_inner_dims(layer, min_p, n_cut_points)
        layers_stats[name] = LayerStats(
            inner_dims=inner_dims,
            errors=[],
            params_savings=_record_param_saving(layer, inner_dims),
        )

    return layers_stats


def profile_model(
    evaluator: BaseEvaluator,
    n_cut_points: int,
    base_loss: float,
    min_p: float,
    *,
    sample_p_grid: bool = False,
) -> dict[str, LayerStats]:
    """
    Profile a model's layers to compute compression statistics.

    Args:
        evaluator: The evaluator instance containing the model to profile.
        n_cut_points: Number of cut points for sampling inner dimensions.
        base_loss: The baseline loss value for comparison.
        min_p: Minimum parameter ratio for layer compression.
        sample_p_grid: If True, use inner dimensions from p-grid; otherwise use standard sampling.

    Returns:
        A dictionary mapping layer names to their computed LayerStats.
    """
    layers_stats = {}
    pbar = tqdm(
        [(n, l) for n, l in evaluator.model.named_modules() if _check_lin_conv(l)],
        leave=False,
        disable=not evaluator.distr.is_local_main_process,
    )

    for name, layer in pbar:
        pbar.set_description(f"Profiling layer: {name}")
        if sample_p_grid:
            inner_dims = _sample_inner_dims_from_p_grid(layer, min_p, n_cut_points)
        else:
            inner_dims = _sample_inner_dims(layer, min_p, n_cut_points)
        layers_stats[name] = _profile_layer(layer, evaluator, inner_dims, base_loss)

    return layers_stats


def _probe_layer_loss(
    evaluator: BaseEvaluator,
    layer: ODLayer,
    inner_dim: int,
    full_loss: float,
) -> float:
    prev_inner_dim = layer.inner_dim
    try:
        layer.inner_dim = inner_dim
        return max(float(evaluator.evaluate()["eval_loss"] - full_loss), 0.0)
    finally:
        layer.inner_dim = prev_inner_dim


def _select_probe_ranks(
    inner_dims: list[int],
    probe_points: int,
) -> list[int]:
    non_full_inner_dims = inner_dims[:-1]
    num_probes = min(probe_points, len(non_full_inner_dims))
    assert num_probes >= 1, "fast profiling requires at least one probe point"

    probe_indices = np.linspace(0, len(non_full_inner_dims) - 1, num_probes, dtype=int)
    probe_inner_dims = [non_full_inner_dims[i] for i in probe_indices]
    return probe_inner_dims


def _tail_energy_by_rank(layer: ODLayer) -> torch.Tensor:
    tail_energy = torch.cumsum(layer.eigs.square().flip(0), dim=0).flip(0)
    return torch.cat((tail_energy, tail_energy.new_zeros(1)))


def _estimate_probe_errors(
    layer: ODLayer,
    inner_dims: list[int],
    tail_by_rank: torch.Tensor,
    probe_ranks: list[int],
    probe_losses: list[float],
) -> list[float]:
    anchor_ranks = probe_ranks + [layer.max_inner_dim]
    anchor_losses = probe_losses + [0.0]
    anchor_tail = tail_by_rank[anchor_ranks].cpu().numpy()
    max_anchor_tail = anchor_tail[0]
    if max_anchor_tail <= 0:
        logger.warning("All singular values are zero")
        return [0.0 for _ in inner_dims]

    # Tail energy decreases as retained rank grows. Reverse for np.interp, which expects
    # an increasing x-axis.
    interp_x = anchor_tail[::-1]
    interp_y = np.array(anchor_losses[::-1], dtype=float)
    inner_dim_tail = tail_by_rank[inner_dims].cpu().numpy()
    return np.interp(inner_dim_tail, interp_x, interp_y).tolist()


def _fast_profile_layer(
    layer: ODLayer,
    evaluator: BaseEvaluator,
    inner_dims: list[int],
    probe_points: int,
    full_loss: float,
) -> LayerStats:
    params_savings = _record_param_saving(layer, inner_dims)
    tail_by_rank = _tail_energy_by_rank(layer)
    probe_ranks = _select_probe_ranks(inner_dims, probe_points)
    probe_losses = [_probe_layer_loss(evaluator, layer, r, full_loss) for r in probe_ranks]
    errors = _estimate_probe_errors(
        layer,
        inner_dims,
        tail_by_rank,
        probe_ranks,
        probe_losses,
    )

    return LayerStats(inner_dims, errors, params_savings)


def _check_eigs(layers: list[tuple[str, ODLayer]]):
    assert all(l.eigs is not None for _, l in layers), "fast profiling requires eigenvalues"


def fast_profile_model(
    evaluator: BaseEvaluator,
    min_p: float,
    n_cut_points: int,
    base_loss: float,
    probe_points: int = 1,
) -> dict[str, LayerStats]:
    """
    Profile model layers efficiently by sampling inner dimensions and measuring performance impact.

    Args:
        evaluator: The model evaluator used for profiling layers.
        min_p: Minimum proportion for sampling inner dimensions.
        n_cut_points: Number of cut points for dimension sampling.
        base_loss: Baseline loss value for comparison.
        probe_points: Number of probe points to evaluate per layer. Defaults to 1.

    Returns:
        Dictionary mapping layer names to their corresponding LayerStats.
    """
    layers_stats = {}
    layers = [(n, l) for n, l in evaluator.model.named_modules() if _check_lin_conv(l)]
    _check_eigs(layers)
    pbar = tqdm(layers, leave=False, disable=not evaluator.distr.is_local_main_process)

    for name, layer in pbar:
        pbar.set_description(f"Fast profiling ({probe_points=}) layer: {name}")
        inner_dims = _sample_inner_dims(layer, min_p, n_cut_points)
        layers_stats[name] = _fast_profile_layer(
            layer, evaluator, inner_dims, probe_points, base_loss
        )

    return layers_stats
