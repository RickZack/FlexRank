"""Coverage for profile data containers, registries, and solution conversion."""

# pylint: disable=missing-function-docstring

import pytest

from flexrank.profiles import (
    DPSearchAlgo,
    DPSolution,
    MergedProfilesData,
    ProfilesData,
    get_profile_search_algo_class_by_name,
    get_profile_solution_class_by_name,
    get_registered_profile_search_algos,
    get_registered_profile_solutions,
    get_sol_from_dict,
)
from flexrank.profiles.profiling import LayerStats, profile_size_only
from flexrank.profiles.utils import (
    count_od_model_params,
    get_od_layers,
    inner_dims_profile_to_params,
)


def test_profile_registries_lookup_and_errors():
    assert get_profile_search_algo_class_by_name("DPSearchAlgo") is DPSearchAlgo
    assert get_profile_solution_class_by_name("DPSolution") is DPSolution
    assert "DPSearchAlgo" in get_registered_profile_search_algos()
    assert "DPSolution" in get_registered_profile_solutions()

    with pytest.raises(ValueError, match="Profile search algo class 'MissingAlgo'"):
        get_profile_search_algo_class_by_name("MissingAlgo")
    with pytest.raises(ValueError, match="Profile solution class 'MissingSolution'"):
        get_profile_solution_class_by_name("MissingSolution")


def test_profiles_data_selection_export_and_merge():
    pred = ProfilesData(
        layers_name=["linear"],
        profiles=[[3], [2], [1]],
        params=[30, 20, 10],
        metrics=[{"eval_loss": 0.1}, {"eval_loss": 0.2}, {"eval_loss": 0.3}],
    )
    calib = ProfilesData(
        layers_name=["linear"],
        profiles=[[3], [2], [1]],
        params=[30, 20, 10],
        metrics=[{"eval_loss": 0.15}, {"eval_loss": 0.25}, {"eval_loss": 0.35}],
    )
    evaluated = ProfilesData(
        layers_name=["linear"],
        profiles=[[3], [2], [1]],
        params=[30, 20, 10],
        metrics=[{"eval_accuracy": 1.0}, {"eval_accuracy": 0.9}, {"eval_accuracy": 0.8}],
    )

    assert pred.max_params == 30
    assert pred.min_params == 10
    assert pred.get_profile_for_params(21) == ([2], 20)
    assert pred.to_export_dict() == {
        "layers_name": ["linear"],
        "profiles": [[3], [2], [1]],
        "params": [30, 20, 10],
    }
    with pytest.raises(ValueError, match="No profile found"):
        pred.get_profile_for_params(5)

    merged = MergedProfilesData.from_profiles_data(pred, calib, evaluated)
    assert merged.pred_metrics == pred.metrics
    assert merged.calib_metrics == calib.metrics
    assert merged.eval_metrics == evaluated.metrics

    with pytest.raises(AssertionError, match="Profiles must match"):
        MergedProfilesData.from_profiles_data(
            pred,
            ProfilesData(["linear"], [[3]], [30], []),
            evaluated,
        )


def test_dp_solutions_convert_to_profiles_data():
    layers_stats = {
        "linear": LayerStats(
            inner_dims=[3, 2, 1],
            errors=[0.0, 0.2, 0.7],
            params_savings=[0, 10, 20],
        ),
        "conv": LayerStats(
            inner_dims=[3, 2],
            errors=[0.0, 0.4],
            params_savings=[0, 15],
        ),
    }
    dp_solution = DPSolution(
        dp_l=[(0.0, [0, 0]), (0.4, [10, 15])],
        layers_stats=layers_stats,
        base_loss=1.0,
        base_model_params=100,
    )

    profiles = dp_solution.to_profiles_data()
    assert profiles.layers_name == ["linear", "conv"]
    assert profiles.profiles == [[3, 3], [2, 2]]
    assert profiles.params == [100, 75]
    assert profiles.metrics == [{"eval_loss": 1.0}, {"eval_loss": 1.4}]

    thresholded = dp_solution.thresholded([1, 20])
    assert thresholded.profiles == [[2, 2]]
    assert thresholded.params == [75]

    restored = get_sol_from_dict(
        "DPSolution",
        {
            "dp_l": [(0.0, [0, 0])],
            "layers_stats": layers_stats,
            "base_loss": 1.0,
            "base_model_params": 100,
        },
    )
    assert isinstance(restored, DPSolution)


def test_profile_size_only_and_param_utilities(tiny_od_model):
    stats = profile_size_only(tiny_od_model, n_cut_points=3, min_p=0.5)

    assert set(stats) == {"linear", "conv"}
    assert stats["linear"].inner_dims[0] == 1
    assert stats["linear"].inner_dims[-1] == tiny_od_model.linear.max_inner_dim
    assert stats["linear"].errors == []

    od_layers = get_od_layers(tiny_od_model)
    full_profile = [layer.max_inner_dim for layer in od_layers]
    assert count_od_model_params(tiny_od_model) == inner_dims_profile_to_params(
        full_profile,
        od_layers,
    )
