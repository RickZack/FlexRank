"""Base abstractions for profile-search algorithms and solutions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from flexrank.profiles.profiling import ProfilesData
from flexrank.trainers.base_trainer import AbstractEvaluator
from flexrank.utils.logger import timed, init_logger

from .registry import register_profile_search_algo, register_profile_solution

logger = init_logger(__name__)

__all__ = ["ProfileSearchAlgo", "ProfileAlgoSolution"]


@register_profile_solution
class ProfileAlgoSolution(ABC):
    """Base class for solutions returned by profile search algorithms."""

    @abstractmethod
    def to_profiles_data(self, *, eager_pruning: bool = False) -> ProfilesData:
        """Converts the internal representation of the solution to a ProfileData"""

    @abstractmethod
    def thresholded(
        self, param_thresholds: list[int], *, eager_pruning: bool = False
    ) -> ProfilesData:
        """Return first profile that reaches each saved-params threshold."""


@dataclass
@register_profile_search_algo
class ProfileSearchAlgo(ABC):
    """Base class for profile search algorithms."""

    evaluator: AbstractEvaluator
    n_models: int
    min_p: float

    @timed(name="Searching profiles", logger=logger)
    def solve(self) -> ProfileAlgoSolution:
        """Run the profile search and return a solution object."""
        return self._solve()

    @abstractmethod
    def _solve(self) -> ProfileAlgoSolution: ...
