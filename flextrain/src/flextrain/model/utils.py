"""Utilities for loading models and normalizing module implementations."""

from typing import Callable, NamedTuple, Optional, TypeAlias, Union, cast

import torch
from transformers import AutoImageProcessor, AutoTokenizer, models
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.trainer_utils import EvalPrediction

from flexrank.utils import init_logger
from flextrain.utils import make_model_contiguous
from flextrain.utils.args import NLPModelArguments, TaskName, VisionModelArguments

from .hf_cv_model import load_cv_model_from_hf, get_cv_metrics
from .hf_nlp_model import load_nlp_model_from_hf

__all__ = ["load_model_from_hf", "set_trainable_layers", "replace_conv1d_with_linear"]

log = init_logger(__name__)

ProcessorType: TypeAlias = Union[AutoTokenizer, AutoImageProcessor]


class ModelWithProcessorAndMetric(NamedTuple):
    """Bundle a model with its processor and optional metrics callback."""

    model: _BaseAutoModelClass
    proc: ProcessorType
    metric_fn: Optional[Callable[[EvalPrediction], dict]]


def load_model_from_hf(
    task: TaskName,
    model_args: NLPModelArguments | VisionModelArguments,
) -> ModelWithProcessorAndMetric:
    """
    Load a model and processor from Hugging Face based on the task type.

    Args:
        task (TaskName): The type of task (NLP or Vision).
        model_args (NLPModelArguments | VisionModelArguments): Model arguments
            containing configuration for loading the model. Should be
            NLPModelArguments if task is TaskName.nlp, otherwise
            VisionModelArguments.

    Returns:
        tuple: A tuple containing:
            - model: The loaded pretrained model
            - processor: The model's processor/tokenizer
            - metrics_fn (Callable | None): The metrics function for evaluation.
              Returns None for NLP tasks, and get_cv_metrics for Vision tasks.

    Raises:
        ValueError: If the specified model cannot be found or loaded from Hugging Face.
    """
    if task is TaskName.NLP:
        args = cast(NLPModelArguments, model_args)
        model, processor = load_nlp_model_from_hf(
            model_name_or_path=args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            load_pretrained=args.load_pretrained_model,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
        )
        metrics_fn = None
    else:
        args = cast(VisionModelArguments, model_args)
        model, processor = load_cv_model_from_hf(
            model_name_or_path=args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            num_labels=args.num_labels,
            load_pretrained=args.load_pretrained_model,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
        )
        metrics_fn: Callable[[EvalPrediction], dict] = get_cv_metrics
    make_model_contiguous(model)

    return ModelWithProcessorAndMetric(model, processor, metrics_fn)


def set_trainable_layers(model: torch.nn.Module, layers_to_train: list[str]):
    """
    Set specific layers of a model to be trainable while freezing others.

    This function configures which layers in a neural network model should have
    gradients enabled for training.

    Args:
        model (torch.nn.Module): The PyTorch model to configure.
        layers_to_train (list[str]): List of layer names to enable for training.
            If empty, all model layers will be set to trainable.

    Raises:
        AssertionError: If none of the specified layer names exist in the model.

    Note:
        - If layers_to_train is empty or None, the entire model remains trainable.
        - Layer names should match those returned by model.named_modules().
        - Logs information about which layers are being trained.
    """
    model_layers_names = set(n for n, _ in model.named_modules())
    layers_to_train = set(layers_to_train) or model_layers_names

    layers_names_subset = layers_to_train & model_layers_names
    assert layers_names_subset, f"None of {layers_to_train} are model's layers"

    if layers_to_train == model_layers_names:
        log.info("Training the whole model")
        return

    log.info("Training a subset of layers: %s", layers_to_train)
    model.requires_grad_(False)

    for name, layer in model.named_modules():
        if name in layers_to_train:
            for param in layer.parameters(recurse=False):
                param.requires_grad = True


def _conv1d_to_linear(
    layer: models.gpt2.modeling_gpt2.Conv1D,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.nn.Linear:
    """
    Build a Linear layer from a Conv1D layer with contiguous parameters.

    Args:
        layer (Conv1D): The Conv1D layer to be replaced.
        device (str | torch.device, optional): The device to place the new Linear layer on.
            If None, it will use the same device as the original layer's weights.
        dtype (torch.dtype, optional): The data type for the new Linear layer's weights.
            If None, uses the dtype of the original layer's weights.

    Returns:
        torch.nn.Linear: A new Linear layer that behaves as the original Conv1D layer.
    """
    device = device or layer.weight.device
    dtype = dtype or layer.weight.dtype

    in_features, out_features = layer.nx, layer.nf

    linear = torch.nn.Linear(
        in_features,
        out_features,
        bias=layer.bias is not None,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(layer.weight.T.contiguous())
        if layer.bias is not None:
            linear.bias.copy_(layer.bias.contiguous())
    return linear


def replace_conv1d_with_linear(
    model: torch.nn.Module,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    """
    Recursively replaces all Conv1D layers in a model with Linear layers.

    Args:
        model (torch.nn.Module): The model in which to replace Conv1D layers.
        device (str | torch.device, optional): The device to place the new Linear layers on.
            If None, it will use the same device as the original layer's weights.
        dtype (torch.dtype, optional): The data type for the new Linear layers' weights.
            Defaults to torch.float32.

    Returns:
        torch.nn.Module: The modified model with Conv1D layers replaced by Linear layers.
    """
    for name, module in model.named_children():
        if isinstance(module, models.gpt2.modeling_gpt2.Conv1D):
            new_module = _conv1d_to_linear(module, device=device, dtype=dtype)
            make_model_contiguous(new_module)
            setattr(model, name, new_module)
        else:
            replace_conv1d_with_linear(module, device=device, dtype=dtype)

    return model
