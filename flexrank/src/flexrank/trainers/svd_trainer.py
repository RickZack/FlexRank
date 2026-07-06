"""SVD-based decomposition trainer."""

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypeVar

import torch
from torch import Tensor
from torch.distributed import broadcast
from torch.nn import Module
from tqdm import tqdm, trange

from flexrank.layers.decomposition import decompose
from flexrank.layers.gram import collect_model_grams
from flexrank.layers.utils import get_decomposition_layers, replace_layer_by_name
from flexrank.profiles.utils import count_od_model_params
from flexrank.trainers.base_trainer import AbstractEvaluator, AbstractTrainer, TrainOutput
from flexrank.types import SVDType
from flexrank.types.distributed import DistributedInfo, get_single_worker_distributed_info
from flexrank.utils.logger import init_logger

logger = init_logger(__name__)

__all__ = ["DecompositionOutput", "SVDType", "SVDTrainer", "SVDTrainingArguments"]


@dataclass(frozen=True)
class SVDTrainingArguments:
    """Training arguments for SVDTrainer."""

    svd_type: SVDType = SVDType.SVD
    exclude_layers_names: list[str] = field(default_factory=list)
    max_svd_data_count: Optional[int] = None
    gram_cache_device: Literal["cpu", "cuda"] = "cpu"
    distr: DistributedInfo = field(default_factory=get_single_worker_distributed_info)

    def __post_init__(self):
        """Validate trainer configuration."""
        if not isinstance(self.svd_type, SVDType):
            self.svd_type = SVDType(self.svd_type)
        if self.svd_type is SVDType.SVD and self.max_svd_data_count:
            raise ValueError("max_svd_data_count is set but svd_type=svd")
        if self.max_svd_data_count is not None and self.max_svd_data_count <= 0:
            raise ValueError("max_svd_data_count must be > 0")


@dataclass(frozen=True)
class DecompositionOutput(TrainOutput):
    """HF-like train output extended with decomposed layer payload."""

    decomposed_layers: list[str]


def _remove_fsdp_name(fsdp_layer_name: str):
    """Strip the FSDP wrapper prefix from a module path, if present."""
    fsdp_header = "_fsdp_wrapped_module."
    return fsdp_layer_name.replace(fsdp_header, "")


_T = TypeVar("_T")


def _flatten_list(nested_list: list[list[_T]]) -> list[_T]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in nested_list for item in sublist]


@dataclass
class SVDTrainer(AbstractTrainer):
    """Trainer that decomposes model's layers using SVD, optionally conditioned on data.
    model attribute is the model whose layers should be updated, evaluator is a BaseEvaluator
    to compute activations for DataSVD. They're are assumed to have the same weights.
    """

    model: Module
    evaluator: AbstractEvaluator
    args: SVDTrainingArguments = field(default_factory=SVDTrainingArguments)

    @property
    def distr(self) -> DistributedInfo:
        """Distributed setup of the trainer."""
        return self.args.distr

    @property
    def eval_model(self) -> Module:
        """Return the model instance used for activation collection."""
        # Soft dependency on HF Accelerator - try to remove it later
        acc = getattr(self.evaluator, "accelerator", None)
        if acc is not None:
            return acc.unwrap_model(self.evaluator.model)
        return self.evaluator.model

    @staticmethod
    def layers_to_decompose(model: Module, exclude_layers_names: list[str]) -> list[str]:
        """List decomposable layer names for the given model."""
        return [name for name, _ in get_decomposition_layers(model, exclude_layers_names)]

    def _select_decomposition_layers(self) -> list[list[str]]:
        """Split the decomposable layers across workers when needed."""
        args = self.args
        layers = self.layers_to_decompose(self.eval_model, args.exclude_layers_names)
        layers = [_remove_fsdp_name(name) for name in layers]
        if args.distr.is_fsdp or args.distr.num_processes == 1:
            return [layers]

        shards = args.distr.num_processes
        chunk_len = math.ceil(len(layers) / shards)
        step = (len(layers) - chunk_len) / (shards - 1)

        chunks = []
        for i in range(shards):
            start = round(i * step)
            chunk = layers[start : start + chunk_len]
            chunks.append(chunk)

        return chunks

    def _collect_layer_inputs(self, names: list[str]):
        """Collect the DataSVD inputs for a given subset of layers."""
        args = self.args
        if args.svd_type is SVDType.SVD:
            return {}

        named_layers = [(name, self.eval_model.get_submodule(name)) for name in names]
        cache_device = (
            args.distr.device if args.gram_cache_device == "cuda" else torch.device("cpu")
        )

        return collect_model_grams(
            named_layers,
            self.evaluator,
            max_data_count=args.max_svd_data_count,
            gram_cache_device=cache_device,
            state=args.distr,
        )

    def _broadcast_layer(self, layer: Module, src: int):
        """Broadcast one decomposed layer's parameters from the source worker."""
        for param in layer.parameters():
            param.data = param.contiguous().data
            broadcast(param.data, src=src)

    def _synchronize_decomposition(self, all_decomp_layers: list[list[tuple[str, Module]]]):
        """Broadcast worker-local decompositions and install them into `self.model`."""
        distr = self.args.distr
        only_one_chunk = len(all_decomp_layers) == 1
        num_layers = sum(map(len, all_decomp_layers))
        pbar = trange(num_layers, disable=not distr.is_local_main_process, leave=False)

        for chunk_no, layers_chunk in enumerate(all_decomp_layers):
            for name, layer in layers_chunk:
                if not only_one_chunk:
                    pbar.set_description(f"Broadcasting layer {name}")
                    self._broadcast_layer(layer, src=chunk_no)
                replace_layer_by_name(self.model, name, layer)
                pbar.update()

    def _init_decomposed_layers(self, layers: list[list[str]]) -> list[list[tuple[str, Module]]]:
        """Run decomposition for each worker-owned layer chunk."""

        def should_skip(idx: int):
            one_chunk = len(layers) == 1
            is_worker_chunk = idx == self.args.distr.process_index
            if one_chunk:
                return False
            return not is_worker_chunk

        all_layers = _flatten_list(layers)
        layer_inputs = self._collect_layer_inputs(all_layers)

        all_decomposed_layers = []
        for chunk_no, layers_chunk in enumerate(layers):
            skip_decomposition = should_skip(chunk_no)
            new_layers = self._decompose_layers(layers_chunk, layer_inputs, skip_decomposition)
            all_decomposed_layers.append(new_layers)

        return all_decomposed_layers

    def _decompose_layers(
        self,
        local_layers: list[str],
        inputs_by_name: dict[str, Tensor],
        skip_decomposition: bool,
    ) -> list[tuple[str, Module]]:
        args = self.args
        decomp_layers: list[tuple[str, Module]] = []
        pbar = tqdm(local_layers, position=args.distr.process_index, disable=skip_decomposition)
        for name in pbar:
            pbar.set_description(
                f"{args.svd_type} (worker {args.distr.process_index}): decomposing layer {name}"
            )
            inputs = inputs_by_name.get(name)

            layer = self.model.get_submodule(name)
            decomp_layer = decompose(
                layer,
                name,
                inputs,
                args.distr,
                skip=skip_decomposition,
            )
            decomp_layers.append((name, decomp_layer))

        return decomp_layers

    @staticmethod
    def _check_train_args(**kwargs: Any) -> None:
        """Warn about HF-style train arguments that this trainer ignores."""
        ignored = {name: value for name, value in kwargs.items() if value is not None}
        for name, value in ignored.items():
            logger.warning("SVDTrainer.train() ignores `%s=%r`.", name, value)

    def train(
        self,
        *args,
        resume_from_checkpoint: str | bool | None = None,
        trial=None,
        ignore_keys_for_eval: list[str] | None = None,
        **kwargs,
    ):
        """Run SVD/DataSVD decomposition and replace layers in `self.model`."""
        if args:
            raise TypeError(f"SVDTrainer.train() got unexpected positional args: {args!r}")
        self._check_train_args(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"SVDTrainer.train() got unexpected kwargs: {unknown}")
        distr = self.args.distr
        if distr.is_distributed and distr.is_local_main_process:
            logger.info(
                "Distributing SVD decomposition on %d processes",
                distr.num_processes,
            )

        layers = self._select_decomposition_layers()
        all_decomp_layers = self._init_decomposed_layers(layers)
        self._synchronize_decomposition(all_decomp_layers)
        flat_layers = _flatten_list(all_decomp_layers)
        return DecompositionOutput(
            global_step=0,
            training_loss=0.0,
            metrics={},
            decomposed_layers=[name for name, _ in flat_layers],
        )

    @property
    def num_model_parameters(self) -> int:
        """Return the effective number of model parameters after decomposition."""
        return self.num_train_model_parameters

    @property
    def num_train_model_parameters(self) -> int:
        """Return the number of trainable parameters in decomposed layers."""
        n_params = count_od_model_params(self.model)
        assert n_params > 0, (
            "No decomposed layers found in the model to count trainable parameters."
        )
        return n_params
