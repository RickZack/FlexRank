"""Helpers for locating, building, and replacing decomposable layers in a model."""

from torch import nn

from .decomposition import build_decomp_like


def replace_layer_by_name(model: nn.Module, layer_name: str, new_layer: nn.Module):
    """Replace a nested layer by dotted module path."""

    def recursive_replace(module, names):
        if len(names) == 1:
            if hasattr(module, names[0]):
                setattr(module, names[0], new_layer)
                return True
            return False

        name = names[0]
        if name.isdigit():
            name = int(name)
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                if name < len(module):
                    return recursive_replace(module[name], names[1:])
                return False

        child = getattr(module, name, None)
        if child is None:
            return False
        return recursive_replace(child, names[1:])

    names = layer_name.split(".")
    if not recursive_replace(model, names):
        raise ValueError(f"Layer '{layer_name}' not found in the model.")


def get_decomposition_layers(
    model: nn.Module, layers_to_exclude: list[str]
) -> list[tuple[str, nn.Module]]:
    """Return all linear and convolution layers eligible for decomposition."""

    def to_be_included(name: str, layer: nn.Module) -> bool:
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            return False
        for excl_name in layers_to_exclude:
            if excl_name in name:
                return False
        return True

    return [(name, layer) for name, layer in model.named_modules() if to_be_included(name, layer)]


def parameterize_model_for_decomposition(
    model: nn.Module,
    layers_to_exclude: list[str] | None = None,
) -> nn.Module:
    """Replace supported dense layers with randomly initialized decomposed layers.

    This is useful before loading a cached decomposed state dict into a freshly
    constructed dense model instance.
    """

    exclude = layers_to_exclude or []
    for name, layer in get_decomposition_layers(model, exclude):
        replace_layer_by_name(model, name, build_decomp_like(layer))
    return model
