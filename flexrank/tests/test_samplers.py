"""Coverage for profile samplers and sampler registry behavior."""

# pylint: disable=missing-function-docstring

import pytest
import torch

from flexrank.layers.base import ODImpl
from flexrank.samplers import (
    AllLayerIndependentRandomSampler,
    AllLayerLinearSampler,
    BaseSampler,
    PredefinedModelsSampler,
    SingleLayerLinearSampler,
    SingleLayerPowerTwoSampler,
    get_registered_sampler_classes,
    get_sampler_class_by_name,
)
from flexrank.samplers.base_sampler import DeployMode


def _max_profile(model: torch.nn.Module) -> tuple[int, ...]:
    return tuple(layer.max_inner_dim for layer in (model.linear, model.conv))


def test_sampler_registry_lookup_and_errors():
    registered = get_registered_sampler_classes()

    assert get_sampler_class_by_name("BaseSampler") is BaseSampler
    assert "SingleLayerLinearSampler" in registered
    with pytest.raises(ValueError, match="Sampler class 'MissingSampler' not found"):
        get_sampler_class_by_name("MissingSampler")


def test_base_sampler_applies_and_resets_profiles(tiny_od_model):
    sampler = BaseSampler(tiny_od_model, samples=[[1, 2]])

    assert sampler.num_od_layers == 2
    assert sampler(return_samples=True) == (3, 3)
    assert tiny_od_model.linear.inner_dim == tiny_od_model.linear.max_inner_dim

    sampler.set_p_for_layers([1, 2])
    assert (tiny_od_model.linear.inner_dim, tiny_od_model.conv.inner_dim) == (1, 2)

    sampler.reset_to_full()
    assert (tiny_od_model.linear.inner_dim, tiny_od_model.conv.inner_dim) == _max_profile(
        tiny_od_model
    )


def test_deploy_modes_prune_or_fallback_as_expected(tiny_od_model):
    sampler = BaseSampler(tiny_od_model)
    sampler.set_p_for_layers([2, 2], deploy_mode=DeployMode.SVD)

    assert tiny_od_model.linear.max_inner_dim == 2
    assert tiny_od_model.conv.max_inner_dim == 2
    assert tiny_od_model.linear.impl is ODImpl.DENSE
    assert tiny_od_model.conv.impl is ODImpl.DENSE

    gar_model = type(tiny_od_model)()
    gar_sampler = BaseSampler(gar_model)
    gar_sampler.set_p_for_layers([2, 2], deploy_mode=DeployMode.GAR)

    assert gar_model.linear.impl is ODImpl.GAR
    assert gar_model.conv.impl is ODImpl.DENSE
    assert gar_model.linear.weight_v.shape == (1, 2)
    assert gar_model.conv.max_inner_dim == 2


def test_structural_sampler_samples_are_within_expected_sets(tiny_od_model):
    single = SingleLayerLinearSampler(
        min_p=0.5,
        num_models=2,
        model=tiny_od_model,
    )
    assert sorted(single._samples) == sorted([[2, 3], [3, 3], [3, 2], [3, 3]])

    power = SingleLayerPowerTwoSampler(model=tiny_od_model)
    assert sorted(power._samples) == sorted([[2, 3], [3, 3], [3, 2], [3, 3]])

    all_layer = AllLayerLinearSampler(
        min_p=0.5,
        num_models=2,
        model=tiny_od_model,
        sandwich=True,
    )
    assert sorted(all_layer._samples) == sorted([[3, 3], [2, 2]])
    gen = all_layer.sampler()
    first_three = [next(gen), next(gen), next(gen)]
    assert first_three[0] == [3, 3]
    assert first_three[2] == [2, 2]

    independent = AllLayerIndependentRandomSampler(
        min_p=0.5,
        num_models=3,
        model=tiny_od_model,
    )
    assert independent._get_inner_dims_upto_width(3).tolist() == [2, 2, 3]


def test_predefined_sampler_uses_explicit_profiles(tiny_od_model):
    sampler = PredefinedModelsSampler(
        model=tiny_od_model,
        samples=[[3, 3], [2, 2]],
        sandwich=True,
    )

    assert sampler.largest_model == [3, 3]
    assert sampler.smallest_model == [2, 2]
    gen = sampler.sampler()
    yielded = [next(gen), next(gen), next(gen)]
    assert yielded[0] == [3, 3]
    assert yielded[2] == [2, 2]
