"""Registry utilities for profile search algorithms and solutions."""

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .base import ProfileAlgoSolution, ProfileSearchAlgo

ProfileSearchAlgoClass = TypeVar("ProfileSearchAlgoClass", bound="ProfileSearchAlgo")
ProfileAlgoSolutionClass = TypeVar("ProfileAlgoSolutionClass", bound="ProfileAlgoSolution")

_PROFILE_SEARCH_ALGO_MAPPINGS: dict[str, type["ProfileSearchAlgo"]] = {}
_PROFILE_SOLUTION_MAPPINGS: dict[str, type["ProfileAlgoSolution"]] = {}


def register_profile_search_algo(
    cls: type[ProfileSearchAlgoClass],
) -> type[ProfileSearchAlgoClass]:
    """Decorator to register a profile search algorithm class."""
    _PROFILE_SEARCH_ALGO_MAPPINGS[cls.__name__] = cls
    return cls


def register_profile_solution(
    cls: type[ProfileAlgoSolutionClass],
) -> type[ProfileAlgoSolutionClass]:
    """Decorator to register a profile solution class."""
    _PROFILE_SOLUTION_MAPPINGS[cls.__name__] = cls
    return cls


def get_profile_search_algo_class_by_name(name: str) -> type["ProfileSearchAlgo"]:
    """Return a registered profile search algorithm class by name."""
    if name not in _PROFILE_SEARCH_ALGO_MAPPINGS:
        available_classes = sorted(_PROFILE_SEARCH_ALGO_MAPPINGS)
        raise ValueError(
            f"Profile search algo class '{name}' not found. Available classes: {available_classes}"
        )
    return _PROFILE_SEARCH_ALGO_MAPPINGS[name]


def get_profile_solution_class_by_name(name: str) -> type["ProfileAlgoSolution"]:
    """Return a registered profile solution class by name."""
    if name not in _PROFILE_SOLUTION_MAPPINGS:
        available_classes = sorted(_PROFILE_SOLUTION_MAPPINGS)
        raise ValueError(
            f"Profile solution class '{name}' not found. Available classes: {available_classes}"
        )
    return _PROFILE_SOLUTION_MAPPINGS[name]


def get_registered_profile_search_algos() -> dict[str, type["ProfileSearchAlgo"]]:
    """Return a copy of the profile search algorithm registry."""
    return dict(_PROFILE_SEARCH_ALGO_MAPPINGS)


def get_registered_profile_solutions() -> dict[str, type["ProfileAlgoSolution"]]:
    """Return a copy of the profile solution registry."""
    return dict(_PROFILE_SOLUTION_MAPPINGS)
