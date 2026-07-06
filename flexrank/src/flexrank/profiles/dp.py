"""Dynamic-programming profile search for layer-wise parameter savings."""

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
from tqdm import trange

from flexrank.profiles.base import ProfileAlgoSolution, ProfileSearchAlgo
from flexrank.profiles.profiling import (
    LayerStats,
    ProfilesData,
    fast_profile_model,
    profile_model,
)
from flexrank.profiles.registry import (
    register_profile_search_algo,
    register_profile_solution,
)
from flexrank.profiles.utils import count_od_model_params
from flexrank.utils.logger import init_logger, timed

__all__ = ["minimize_error_with_any_drops", "DPSolution", "DPSearchAlgo"]

logger = init_logger(__name__)


# ===================================
# Base Dynamic Programming algorithm
# ===================================

SavingsErrors = tuple[list[int], list[float]]
LayersSavingsErrors = list[SavingsErrors]


class DPAlgoStepResult(NamedTuple):
    """Frontier states kept after each DP layer update."""

    total_savings: np.ndarray
    total_errors: np.ndarray
    prev_state_idx: Optional[np.ndarray] = None
    chosen_savings: Optional[np.ndarray] = None


@dataclass
class BackpointerSolution:
    """Backpointers needed to reconstruct per-layer DP choices."""

    _l: list[tuple] = field(default_factory=list)

    def update(self, pointers: np.ndarray, savings: np.ndarray):
        """Append backpointers for one processed layer."""
        self._l.append((pointers, savings))

    def full_solution(self, savings: np.ndarray, errors: np.ndarray):
        """Reconstruct all finite DP solutions from stored backpointers."""
        return _reconstruct_full_solution(self._l, savings, errors)


def _get_layer_saving_errors(
    savings_errors: SavingsErrors,
) -> tuple[np.ndarray, np.ndarray]:
    savings, errors = savings_errors
    savings = np.array(savings, dtype=int)
    errors = np.array(errors, dtype=float)

    return savings, errors


def _generate_layer_candidates(
    frontier: DPAlgoStepResult, layer_savings: np.ndarray, layer_errors: np.ndarray
) -> DPAlgoStepResult:
    n_states = frontier.total_savings.size
    n_opts = layer_savings.size

    # Calculate all new possibilities given:
    # (i) previous candidates and (ii) (savings,errors) for current layer
    cand_savings = (frontier.total_savings[:, None] + layer_savings[None, :]).ravel()
    cand_errors = (frontier.total_errors[:, None] + layer_errors[None, :]).ravel()

    # For each new candidate (saving, error), track the candidate it has been generated from
    prev_state_idx = np.repeat(np.arange(n_states, dtype=int), n_opts)
    # What saving option used at this layer
    chosen_savings = np.tile(layer_savings, n_states)

    # Add "no saving" option (just carry on previous candidates)
    cand_savings = np.concatenate([cand_savings, frontier.total_savings])
    cand_errors = np.concatenate([cand_errors, frontier.total_errors])
    prev_state_idx = np.concatenate([prev_state_idx, np.arange(n_states)])
    chosen_savings = np.concatenate([chosen_savings, np.zeros(n_states, dtype=int)])

    return DPAlgoStepResult(cand_savings, cand_errors, prev_state_idx, chosen_savings)


def _keep_min_error_candidates(frontier: DPAlgoStepResult) -> DPAlgoStepResult:
    # The idea is to prune out suboptimal candidates
    # To do so, for each saving, we keep the candidate with min error
    # To have a faster vectorized implementation (without for-loops) we sort by (saving,error)
    # and take the first candidate of each block
    order = np.lexsort((frontier.total_errors, frontier.total_savings))
    s = frontier.total_savings[order]
    e = frontier.total_errors[order]
    prev_state_idx = frontier.prev_state_idx[order]
    chosen_savings = frontier.chosen_savings[order]

    # First occurrence per unique key = minimal error (because sorted by error)
    keep = np.r_[True, s[1:] != s[:-1]]
    return DPAlgoStepResult(s[keep], e[keep], prev_state_idx[keep], chosen_savings[keep])


def _prune_domin_candidates(frontier: DPAlgoStepResult) -> DPAlgoStepResult:
    # Pareto pruning within current layer
    # Remove dominated states: if a state has fewer savings but higher error
    order = np.argsort(frontier.total_savings)
    s = frontier.total_savings[order]
    e = frontier.total_errors[order]
    prev_state_idx = frontier.prev_state_idx[order]
    chosen_savings = frontier.chosen_savings[order]

    # Scan from largest saving to smallest; keep only improvements
    rev_e = e[::-1]
    best_right = np.minimum.accumulate(rev_e)
    prev_best = np.empty_like(best_right)
    prev_best[0] = np.inf
    prev_best[1:] = best_right[:-1]
    keep = (rev_e < prev_best)[::-1]

    # Pruned states for current layer
    return DPAlgoStepResult(s[keep], e[keep], prev_state_idx[keep], chosen_savings[keep])


def _filter_nested_solutions(
    solutions: list[tuple[float, list[int]]],
) -> list[tuple[float, list[int]]]:
    """Filter final solutions to a componentwise-nested chain."""
    nested = []
    last_kept = None

    for error, drops in solutions:
        cur_drops = np.array(drops, dtype=int)
        if last_kept is None or np.all(cur_drops >= last_kept):
            nested.append((error, drops))
            last_kept = cur_drops

    return nested


def _reconstruct_full_solution(
    backpointers: list[tuple[np.ndarray, np.ndarray]],
    savings: np.ndarray,
    errors: np.ndarray,
) -> list[tuple[float, np.ndarray]]:
    # Backtracking over the backpointers to reconstruct the full solution
    solutions = []
    num_layers = len(backpointers)
    n_final = savings.size
    for final_idx in range(n_final):
        e = errors[final_idx]
        if not np.isfinite(e):
            continue

        # Walk backwards through layers to reconstruct savings vector
        drops = np.zeros(num_layers, dtype=int)
        curr_idx = final_idx
        for layer in range(num_layers - 1, -1, -1):
            prev_idx_layer, saving_layer = backpointers[layer]
            drops[layer] = int(saving_layer[curr_idx])
            curr_idx = int(prev_idx_layer[curr_idx])
        solutions.append((float(e), drops))

    return solutions


def _filter_best_solutions(
    solutions: list[tuple[float, np.ndarray]],
) -> list[tuple[float, list[int]]]:
    # Extract arrays of saving and errors
    savings = np.array([d for _, d in solutions])
    errors = np.array([e for e, _ in solutions])
    # Total saving (i.e. summed over layers) for each solution
    sums = savings.sum(axis=1)

    # Ordering the solutions according to total saving
    order = np.argsort(sums)
    sums = sums[order]
    errors = errors[order]
    savings = savings[order]

    rev_errs = errors[::-1]
    min_prev = np.minimum.accumulate(rev_errs)
    min_prev_shift = np.empty_like(min_prev)
    min_prev_shift[0] = np.inf
    min_prev_shift[1:] = min_prev[:-1]
    keep_rev = rev_errs < min_prev_shift
    keep_mask = keep_rev[::-1]

    # Convert numpy arrays to plain lists for JSON serialization
    return [
        (float(error), drop_vec.tolist())
        for error, drop_vec in zip(errors[keep_mask], savings[keep_mask])
    ]


def minimize_error_with_any_drops(
    savings_errors: LayersSavingsErrors,
    *,
    nested: bool = True,
):
    """
    Exact dynamic programming for minimizing error with parameter savings.

    Key ideas:
      - Dynamic programming (DP) over layers.
      - Each DP state = (total_saving, min_error_so_far).
      - For each layer, we expand states with all possible savings.
      - Use NumPy to vectorize:
          * Expand all states in parallel
          * Keep only the minimal error per total_saving
      - Apply Pareto pruning at each layer:
          * Remove states that are strictly worse (higher error & fewer savings).
      - Store backpointers so we can reconstruct the full savings vector.
      - Optionally apply an exact nested filter on the final Pareto solutions.

    Returns:
        List of (error, drops_vector) solutions,
        Pareto-filtered, sorted by total_savings.
    """

    num_layers = len(savings_errors)

    # Initial state: 0 saving, 0 error
    frontier = DPAlgoStepResult(
        np.array([0], dtype=int),  # total savings
        np.array([0.0], dtype=float),  # total errors
    )

    # Store backpointers for reconstruction
    # Each element = (prev_idx_for_each_state, saving_for_each_state, keys, errs)
    sol = BackpointerSolution()

    # Build the table of backpointers
    for layer in trange(1, num_layers + 1, desc="Determining ranks via DP", leave=False):
        layer_savings, layer_errors = _get_layer_saving_errors(savings_errors[layer - 1])
        frontier = _generate_layer_candidates(frontier, layer_savings, layer_errors)
        frontier = _keep_min_error_candidates(frontier)
        frontier = _prune_domin_candidates(frontier)

        # Save backpointers for reconstruction
        sol.update(frontier.prev_state_idx, frontier.chosen_savings)

    solutions = sol.full_solution(frontier.total_savings, frontier.total_errors)
    solutions = _filter_best_solutions(solutions)
    if nested:
        return _filter_nested_solutions(solutions)
    return solutions


# ===================================
# DP Solver interface with outside
# ===================================


@register_profile_solution
@dataclass(frozen=True)
class DPSolution(ProfileAlgoSolution):
    """DP search output plus metadata needed to build profiles."""

    dp_l: list[tuple[float, list[int]]]  # (error, param_savings_per_layer)
    layers_stats: dict[str, LayerStats]
    base_loss: float
    base_model_params: int

    def to_profiles_data(self, *, eager_pruning: bool = False) -> ProfilesData:
        p_to_dim = {
            name: dict(zip(stat.params_savings, stat.inner_dims))
            for name, stat in self.layers_stats.items()
        }
        losses = []
        params = []
        od_profiles = []

        profiled_layers = list(self.layers_stats.keys())

        for error, param_saving in self.dp_l:
            od_prof = []
            for name, p_saving in zip(profiled_layers, param_saving):
                od_prof.append(p_to_dim[name][p_saving])
            profile_params = self.base_model_params - sum(param_saving)

            losses.append({"eval_loss": self.base_loss + error})
            params.append(profile_params)
            od_profiles.append(od_prof)

        return ProfilesData(profiled_layers, od_profiles, params, losses)

    def _next_sol_index(
        self,
        cur_idx: int,
        saved_params: int,
        param_thresholds: list[float],
        *,
        eager_pruning: bool,
    ) -> int:
        def should_increment(idx: int):
            return idx < len(param_thresholds) and saved_params >= param_thresholds[idx]

        max_steps = len(param_thresholds) - cur_idx if eager_pruning else 1
        for _ in range(max_steps):
            if should_increment(cur_idx):
                cur_idx += 1
            else:
                break

        return cur_idx

    def thresholded(
        self, param_thresholds: list[int], *, eager_pruning: bool = False
    ) -> ProfilesData:
        n_solutions = len(param_thresholds)
        dp_lf = []
        i = 0
        # We assume param_thresholds is sorted in increasing order
        # We assume dp_L is sorted in increasing order of error
        #  and increasing order of params saved
        # We find the first solution that meets each threshold
        # If solutions also meet higher thresholds, we skip them
        for loss, saved_params in self.dp_l:
            if sum(saved_params) >= param_thresholds[i]:
                i = self._next_sol_index(
                    i,
                    sum(saved_params),
                    param_thresholds,
                    eager_pruning=eager_pruning,
                )
                dp_lf.append((loss, saved_params))
                if i >= n_solutions:
                    break
        thresholded_sol = DPSolution(
            dp_lf, self.layers_stats, self.base_loss, self.base_model_params
        )
        return thresholded_sol.to_profiles_data()


@register_profile_search_algo
@dataclass
class DPSearchAlgo(ProfileSearchAlgo):
    """Profile-search algorithm backed by the exact DP solver."""

    fast_probe_points: int = 0

    def _print_header(self):
        logger.info("\n === RUNNING PROFILE SEARCH VIA DYNAMIC PROGRAMMING ===")

    @staticmethod
    def _correct_error_estimates(
        layers_stats: dict[str, LayerStats],
    ) -> dict[str, LayerStats]:

        def make_strictly_decreasing(errors: list[float], eps: float = 1e-8) -> list[float]:
            # Correct errors estimates, such that DP will find possible to cut ranks
            # without increasing the error. This can occur because the eval loss on the
            # calibration set (which is different from the training set used for pretraining)
            # can be lowe when cutting some ranks. The idea is: "I trust later error estimates
            # (big models) more than earlier ones", as a result, other estimates are moved up
            rev = np.asarray(errors, float)[::-1]
            rev = np.maximum.accumulate(rev)
            errors_up = rev[::-1] + eps * np.arange(len(rev))
            return errors_up.tolist()

        layers_stats = layers_stats.copy()
        for name, s in layers_stats.items():
            errors = make_strictly_decreasing(s.errors)
            layers_stats[name] = LayerStats(s.inner_dims, errors, s.params_savings)
        return layers_stats

    @timed("Layer profiling", logger)
    def _layer_profiling(self):
        base_loss = self.evaluator.evaluate()["eval_loss"]
        base_model_params = count_od_model_params(self.evaluator.model)

        if not self.fast_probe_points:
            layers_stats = profile_model(self.evaluator, self.n_models, base_loss, self.min_p)
            return base_loss, base_model_params, layers_stats

        layers_stats = fast_profile_model(
            self.evaluator, self.min_p, self.n_models, base_loss, self.fast_probe_points
        )
        return base_loss, base_model_params, layers_stats

    @timed("DP algorithm", logger)
    def _solving_via_dp(self, layers_stats: dict[str, LayerStats]):
        # Trick to avoid DP search to cut ranks prematurely
        layers_stats = self._correct_error_estimates(layers_stats)
        savings_errors = [(stat.params_savings, stat.errors) for stat in layers_stats.values()]
        return minimize_error_with_any_drops(savings_errors)

    def _solve(self) -> DPSolution:
        self._print_header()
        base_loss, base_model_params, layers_stats = self._layer_profiling()
        # optimize with dynamic programming
        dp_l = self._solving_via_dp(layers_stats)
        return DPSolution(dp_l, layers_stats, base_loss, base_model_params)
