"""Utilities for collecting layer input Gram matrices."""

from typing import Optional

import torch
import torch.distributed as distr
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from flexrank.trainers.base_trainer import AbstractEvaluator
from flexrank.types import DistributedInfo, get_single_worker_distributed_info
from flexrank.utils import init_logger

log = init_logger(__name__)

__all__ = ["collect_model_grams"]


class GramStat:
    """Running average of a Gram matrix and its row count."""

    gram: torch.Tensor
    n_rows: torch.Tensor
    n_examples: int

    def __init__(self, dim: int, device: torch.device, dtype: torch.dtype):
        self.gram = torch.zeros((dim, dim), device=device, dtype=dtype)
        self.n_rows = torch.zeros(1, device=device, dtype=torch.long)
        self.n_examples = 0

    @staticmethod
    def make_gram(
        matrix: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a batch Gram matrix and its effective row count."""
        matrix = matrix.to(device, dtype)
        batch_gram = matrix.T @ matrix
        rows = torch.tensor([matrix.shape[0]], device=matrix.device, dtype=torch.long)
        return batch_gram, rows

    @staticmethod
    def _reduce_gram(batch_gram: torch.Tensor, batch_rows: torch.Tensor, state: DistributedInfo):
        """Synchronize batch statistics across workers when distributed."""
        if state.is_distributed:
            distr.all_reduce(batch_gram, op=distr.ReduceOp.SUM)
            distr.all_reduce(batch_rows, op=distr.ReduceOp.SUM)

    def update(self, matrix: torch.Tensor, batch_examples: int, state: DistributedInfo):
        """Update the running Gram average with a new batch."""
        batch_gram, batch_rows = GramStat.make_gram(matrix, self.gram.device, self.gram.dtype)
        GramStat._reduce_gram(batch_gram, batch_rows, state)
        self.update_stat(batch_gram, batch_rows)
        self.n_examples += batch_examples

    def update_stat(self, batch_gram: torch.Tensor, batch_examples: torch.Tensor):
        """Update the running average from precomputed statistics."""
        prev_avg_factor = self.n_rows
        new_avg_factor = prev_avg_factor + batch_examples
        mov_avg_factor = prev_avg_factor / new_avg_factor
        self.gram.mul_(mov_avg_factor).add_(batch_gram, alpha=1.0 / new_avg_factor.item())
        self.n_rows = new_avg_factor


class LayersGramStats:  # pylint: disable=too-few-public-methods
    """Container for all tracked layer Gram statistics."""

    grams: dict[str, GramStat]

    def __init__(
        self,
        named_layers: list[tuple[str, nn.Module]],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        def get_in_features(layer: nn.Linear | nn.Conv2d) -> int:
            if isinstance(layer, nn.Linear):
                return layer.in_features
            k_size = layer.kernel_size
            k1, k2 = k_size if isinstance(k_size, tuple) else (k_size, k_size)
            return k1 * k2 * layer.in_channels

        self.grams = {
            n: GramStat(get_in_features(l), device=device, dtype=dtype) for n, l in named_layers
        }

    def to_grams(self) -> dict[str, torch.Tensor]:
        """Extract the raw Gram tensors for all tracked layers."""
        return {n: stat.gram for n, stat in self.grams.items()}


def _prepare_layer_input_matrix(
    layer: nn.Module,
    layer_input: torch.Tensor,
) -> torch.Tensor:
    """Normalize a layer input tensor into a 2D matrix for Gram computation."""

    if isinstance(layer, nn.Conv2d):
        layer_input = F.pad(
            layer_input,
            (layer.padding[1], layer.padding[1], layer.padding[0], layer.padding[0]),
        )
        layer_input = _im2col(layer_input, layer.kernel_size, layer.stride, layer_input.device)

    if layer_input.dim() == 3:
        layer_input = layer_input.reshape(-1, layer_input.shape[-1])

    return layer_input


def _slice_batch_examples(
    layer_input: torch.Tensor,
    total_examples: int,
    max_data_count: Optional[int],
) -> tuple[Optional[torch.Tensor], int, bool]:
    """Clip a raw layer-input batch to the remaining example budget."""
    examples_in_batch = layer_input.shape[0]

    if max_data_count is None:
        return layer_input, examples_in_batch, False

    remaining_examples = max_data_count - total_examples
    if remaining_examples <= 0:
        return None, 0, True

    if examples_in_batch > remaining_examples:
        layer_input = layer_input[:remaining_examples]
        examples_in_batch = layer_input.shape[0]

    if examples_in_batch <= 0:
        return None, 0, True

    stop_requested = total_examples + examples_in_batch >= max_data_count
    return layer_input, examples_in_batch, stop_requested


def _make_collection_pbar(
    desc: str,
    total: Optional[int] = None,
    *,
    unit: str,
    position: int,
    disable: bool = False,
):
    """Create a standardized progress bar for Gram collection."""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=False,
        position=position,
        disable=disable,
    )


def _im2col(input_tensor, kernel_size, stride, _device):
    """
    Convert an image tensor into im2col form using `torch.nn.functional.unfold`.

    Args:
    input_tensor (torch.Tensor): Input tensor of shape
        `(batch_size, in_channels, height, width)`.
    kernel_size (tuple): Convolution kernel size.
    stride (tuple): Convolution stride.
    _device (str): Kept for backward-compatible call sites.

    Returns:
    torch.Tensor: Column matrix of shape `(n_patches, patch_dim)`.
    """
    unfolded = F.unfold(input_tensor, kernel_size=kernel_size, stride=stride)
    return unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1])


def _check_input_args(network, named_layers, max_data_count):
    """Validate Gram collection inputs before hooks are registered."""
    valid_data_count = max_data_count is None or max_data_count > 0
    assert valid_data_count, "max_data_count must be > 0"

    layers_in_network = all(l in network.modules() for _, l in named_layers)
    assert layers_in_network, "named_layers contains layer which are not part of network"


def _get_max_data_count_slice(max_data_count: Optional[int]) -> int:
    """Split a global example budget evenly across distributed workers."""
    is_distributed = distr.is_available() and distr.is_initialized()
    if is_distributed and max_data_count is not None:
        return max(1, max_data_count // distr.get_world_size())
    return max_data_count


def collect_model_grams(  # pylint: disable=too-many-arguments,too-many-locals
    named_layers: list[tuple[str, nn.Module]],
    trainer: AbstractEvaluator,
    *,
    max_data_count: Optional[int] = None,
    gram_cache_device: torch.device = torch.device("cpu"),
    gram_cache_dtype: torch.dtype = torch.float32,
    state: Optional[DistributedInfo] = None,
) -> dict[str, torch.Tensor]:
    """
    Collect Gram matrices for multiple layers in a single evaluation pass.

    Args:
        named_layers: List of `(layer_name, layer_module)` pairs to track.
        trainer: Evaluator-like object exposing `evaluate()`.
        max_data_count: Optional cap on dataset examples consumed per layer.
        gram_cache_device: Device where running Gram matrices are stored.
        gram_cache_dtype: Torch dtype used to store the Gram matrices.
        is_distr_local_main: whether the current process is a local main processes

    Returns:
        Mapping from layer name to its normalized input Gram matrix.
    """
    _check_input_args(trainer.model, named_layers, max_data_count)
    state = state or get_single_worker_distributed_info()
    stats = LayersGramStats(named_layers, gram_cache_device, gram_cache_dtype)
    max_data_count = _get_max_data_count_slice(max_data_count)

    stop_requested = False
    batch_pbar = _make_collection_pbar(
        "Forwarding data batches",
        unit="batch",
        position=0,
        disable=not state.is_local_main_process,
    )
    layer_pbar = _make_collection_pbar(
        "Collecting Gram matrix",
        len(named_layers),
        unit="layer",
        position=1,
        disable=not state.is_local_main_process,
    )

    class _EarlyStopEval(Exception):
        pass

    def make_hook(stat: GramStat, layer_name: str, layer: nn.Module):
        def hook(_module, inputs, _output):
            nonlocal stop_requested

            layer_input, batch_examples, batch_done = _slice_batch_examples(
                inputs[0],
                stat.n_examples,
                max_data_count,
            )
            if layer_input is None:
                stop_requested = True
                return

            input_matrix = _prepare_layer_input_matrix(layer, layer_input)
            stat.update(input_matrix, batch_examples, state)

            layer_pbar.set_description(f"Collecting Gram matrix for {layer_name}")
            layer_pbar.update(1)
            if batch_done:
                stop_requested = True

        return hook

    def stop_hook(_module, _inputs, _output):
        batch_pbar.update(1)
        layer_pbar.reset()
        if stop_requested:
            raise _EarlyStopEval

    handles = [
        layer.register_forward_hook(make_hook(stats.grams[name], name, layer))
        for name, layer in named_layers
    ]
    stop_handle = trainer.model.register_forward_hook(stop_hook)
    try:
        trainer.evaluate()
    except _EarlyStopEval:
        pass
    finally:
        stop_handle.remove()
        for handle in handles:
            handle.remove()
        layer_pbar.close()
        batch_pbar.close()

    missing_layers = [name for name, stat in stats.grams.items() if not stat.n_examples]
    if missing_layers:
        raise RuntimeError(
            "get_inputs_llm_multi did not collect activations for: " + ", ".join(missing_layers)
        )

    return stats.to_grams()
