"""Public model API for loading, wrapping, and model-specific utilities."""

from .dinov3_for_image_classification import DINOv3ViTForImageClassification
from .flexrank_model import FlexRankConfig, FlexRankModel
from .utils import (
    ModelWithProcessorAndMetric,
    load_model_from_hf,
    replace_conv1d_with_linear,
    set_trainable_layers,
)

__all__ = [
    "load_model_from_hf",
    "set_trainable_layers",
    "replace_conv1d_with_linear",
    "ModelWithProcessorAndMetric",
    "DINOv3ViTForImageClassification",
    "FlexRankModel",
    "FlexRankConfig",
]
