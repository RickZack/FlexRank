"""Utilities for profile parameter accounting."""

from flexrank.utils.logger import init_logger
from ..layers.base import ODLayer

logger = init_logger(__name__)

__all__ = [
    "count_od_model_params",
    "get_od_layers",
    "inner_dims_profile_to_params",
]


def _check_lin_conv(layer):
    return isinstance(layer, ODLayer)


def _get_num_params(layer: ODLayer) -> int:
    assert isinstance(layer, ODLayer), f"Expected an ODLayer, but got {type(layer)}"
    return layer.num_parameters


def count_od_model_params(model) -> int:
    """Returns the number of parameters in OD layers"""
    return sum(_get_num_params(layer) for layer in model.modules() if _check_lin_conv(layer))


def get_od_layers(model) -> list[ODLayer]:
    """Return the model's OD layers in module traversal order."""
    return [layer for layer in model.modules() if _check_lin_conv(layer)]


def inner_dims_profile_to_params(inner_dims: list[int], od_layers: list[ODLayer]) -> int:
    """
    Calculate the total number of parameters for a given inner-dimension profile.

    Args:
        inner_dims: A 1D sequence of inner dimensions, one for each layer.
        od_layers: A list of ODLayer objects to be profiled.

    Returns:
        int: The total number of parameters across all layers.
    """
    assert len(inner_dims) == len(od_layers)
    return sum(
        layer.get_num_parameters(inner_dim) for inner_dim, layer in zip(inner_dims, od_layers)
    )
