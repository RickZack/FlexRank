"""Registry utilities for sampler classes."""

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .base_sampler import BaseSampler

SamplerClass = TypeVar("SamplerClass", bound="BaseSampler")

_SAMPLER_CLASS_MAPPINGS: dict[str, type["BaseSampler"]] = {}


def register_sampler(cls: type[SamplerClass]) -> type[SamplerClass]:
    """Decorator to register a sampler class for dynamic instantiation."""
    _SAMPLER_CLASS_MAPPINGS[cls.__name__] = cls
    return cls


def get_sampler_class_by_name(name: str) -> type["BaseSampler"]:
    """Return a registered sampler class by name."""
    if name not in _SAMPLER_CLASS_MAPPINGS:
        available_classes = sorted(_SAMPLER_CLASS_MAPPINGS)
        raise ValueError(
            f"Sampler class '{name}' not found. Available classes: {available_classes}"
        )
    return _SAMPLER_CLASS_MAPPINGS[name]


def get_registered_sampler_classes() -> dict[str, type["BaseSampler"]]:
    """Return a copy of the sampler registry."""
    return dict(_SAMPLER_CLASS_MAPPINGS)
