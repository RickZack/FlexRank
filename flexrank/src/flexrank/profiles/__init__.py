"""Public profiling API for search algorithms, solutions, and profile data."""

from typing import Any

from flexrank.trainers.base_trainer import AbstractEvaluator
from .base import ProfileAlgoSolution, ProfileSearchAlgo
from .dp import DPSearchAlgo, DPSolution
from .profiling import MergedProfilesData, ProfilesData
from .registry import (
    get_profile_search_algo_class_by_name,
    get_profile_solution_class_by_name,
    get_registered_profile_search_algos,
    get_registered_profile_solutions,
    register_profile_search_algo,
    register_profile_solution,
)
from .utils import get_od_layers, inner_dims_profile_to_params

__all__ = [
    "ProfileSearchAlgo",
    "ProfileAlgoSolution",
    "ProfilesData",
    "MergedProfilesData",
    "DPSearchAlgo",
    "DPSolution",
    "register_profile_search_algo",
    "register_profile_solution",
    "get_profile_search_algo_class_by_name",
    "get_profile_solution_class_by_name",
    "get_registered_profile_search_algos",
    "get_registered_profile_solutions",
    "get_profile_search_algo",
    "get_sol_from_dict",
    "get_od_layers",
    "inner_dims_profile_to_params",
]


def get_profile_search_algo(
    cls_name: str, evaluator: AbstractEvaluator, *, algo_kwargs
) -> ProfileSearchAlgo:
    """Create a profile-search algorithm from its class name and evaluator."""
    assert "evaluator" not in algo_kwargs, "Evaluator should not be a keyword argument, overriding"
    cls = get_profile_search_algo_class_by_name(cls_name)
    kwargs = {**algo_kwargs, "evaluator": evaluator}
    return cls(**kwargs)


def get_sol_from_dict(cls_name: str, fields: dict[str, Any]) -> ProfileAlgoSolution:
    """Create a profile solution from its class name and serialized fields."""
    cls = get_profile_solution_class_by_name(cls_name)
    return cls(**fields)
