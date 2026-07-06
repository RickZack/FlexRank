"""Tests for FlexRankModel checkpoint round-tripping."""

import json
import shutil
from types import SimpleNamespace

import pytest

from flexrank.layers.base import ODImpl, ODLayer
from flexrank.profiles import ProfilesData, get_od_layers, inner_dims_profile_to_params
from flexrank.samplers.base_sampler import DeployMode
from flextrain.model import FlexRankModel

transformers = pytest.importorskip("transformers")


def _od_layer_state_max_dims(model):
    return {name: state["max_inner_dim"] for name, state in model.config.od_layer_state.items()}


class _ExportableArgs:
    def __init__(self, **data):
        self._data = data
        for key, value in data.items():
            setattr(self, key, value)

    def to_export_dict(self):
        return dict(self._data)


def _training_args():
    return SimpleNamespace(
        dataset=_ExportableArgs(name="dummy"),
        decomposition=_ExportableArgs(
            freeze_non_decomposed=False,
            exclude_layers_names=["lm_head"],
        ),
        profile_algo=_ExportableArgs(name="dummy"),
        sampler=_ExportableArgs(name="dummy"),
        train=_ExportableArgs(name="dummy"),
        model=_ExportableArgs(name="dummy"),
    )


def test_flexrank_model_from_pretrained_restores_weight_v_and_profiles(tmp_path):
    """FlexRank factor names should load without HF weight-norm key remapping."""
    save_dir = tmp_path / "flexrank-gpt2"
    config = transformers.GPT2Config(
        vocab_size=128,
        n_positions=16,
        n_embd=32,
        n_layer=1,
        n_head=4,
    )
    expected_base_model_config = json.loads(json.dumps(config.to_dict()))
    model = FlexRankModel.from_config(
        config,
        base_auto_class="AutoModelForCausalLM",
        exclude_layers=["lm_head"],
    )

    od_layers = get_od_layers(model.base_model)
    profiles = [
        [max(1, int(layer.max_inner_dim * ratio)) for layer in od_layers]
        for ratio in (1.0, 0.5, 0.25)
    ]
    params = [inner_dims_profile_to_params(profile, od_layers) for profile in profiles]
    model.profiles_data = ProfilesData(
        layers_name=[
            name for name, layer in model.base_model.named_modules() if isinstance(layer, ODLayer)
        ],
        profiles=profiles,
        params=params,
        metrics=[{} for _ in profiles],
    )

    shutil.rmtree(save_dir, ignore_errors=True)
    model.save_pretrained(save_dir)

    saved_config = json.loads((save_dir / "config.json").read_text())
    assert "decomposed_layers" not in saved_config
    assert "n_embd" not in saved_config
    assert saved_config["base_model_config"] == expected_base_model_config
    assert set(saved_config["od_layer_state"]) == set(model.config.od_layer_state)
    assert all(
        state["inner_dim"] == state["max_inner_dim"]
        for state in saved_config["od_layer_state"].values()
    )
    assert "flexrank" in (save_dir / "requirements.txt").read_text()
    copied_source = (save_dir / "flexrank_model.py").read_text()
    assert "from flextrain" not in copied_source
    assert "import flextrain" not in copied_source

    loaded = FlexRankModel.from_pretrained(save_dir)

    assert loaded.config.base_model_config == expected_base_model_config
    assert loaded.profiles_data is not None
    assert loaded.profiles_data.params == params
    assert all(
        key in loaded.state_dict()
        for key in (
            "_wrapped_model.transformer.h.0.attn.c_attn.weight_u",
            "_wrapped_model.transformer.h.0.attn.c_attn.weight_v",
        )
    )


def test_flexrank_model_from_model_keeps_base_config_nested():
    """Wrapping an existing model should not flatten base config fields."""
    config = transformers.GPT2Config(
        vocab_size=128,
        n_positions=16,
        n_embd=32,
        n_layer=1,
        n_head=4,
    )
    model = FlexRankModel.from_config(
        config,
        base_auto_class="AutoModelForCausalLM",
        exclude_layers=["lm_head"],
    )
    decomposed_layer_names = list(model.decomposed_layer_names)
    expected_base_model_config = json.loads(json.dumps(model.base_model.config.to_dict()))

    wrapped = FlexRankModel.from_model(
        model.base_model,
        _training_args(),
        decomposed_layer_names=decomposed_layer_names,
    )
    config_dict = wrapped.config.to_dict()

    assert "n_embd" not in config_dict
    assert json.loads(json.dumps(config_dict["base_model_config"])) == expected_base_model_config


def test_flexrank_model_from_pretrained_restores_physically_pruned_layers(tmp_path):
    """Physically pruned layer dimensions should be serialized for reloading."""
    save_dir = tmp_path / "flexrank-gpt2-pruned"
    config = transformers.GPT2Config(
        vocab_size=128,
        n_positions=16,
        n_embd=32,
        n_layer=1,
        n_head=4,
    )
    model = FlexRankModel.from_config(
        config,
        base_auto_class="AutoModelForCausalLM",
        exclude_layers=["lm_head"],
    )

    od_layers = get_od_layers(model.base_model)
    profiles = [
        [max(1, int(layer.max_inner_dim * ratio)) for layer in od_layers] for ratio in (1.0, 0.5)
    ]
    params = [inner_dims_profile_to_params(profile, od_layers) for profile in profiles]
    model.profiles_data = ProfilesData(
        layers_name=[
            name for name, layer in model.base_model.named_modules() if isinstance(layer, ODLayer)
        ],
        profiles=profiles,
        params=params,
        metrics=[{} for _ in profiles],
    )

    model.reduce_size(size_ratio=0.6, deploy_mode=DeployMode.GAR)
    assert _od_layer_state_max_dims(model) == {
        name: layer.inner_dim
        for name, layer in model.base_model.named_modules()
        if isinstance(layer, ODLayer)
    }

    model.save_pretrained(save_dir)

    saved_config = json.loads((save_dir / "config.json").read_text())
    assert "decomposed_layers" not in saved_config
    assert {
        name: state["max_inner_dim"] for name, state in saved_config["od_layer_state"].items()
    } == _od_layer_state_max_dims(model)
    assert all(state["impl"] == "gar" for state in saved_config["od_layer_state"].values())

    loaded = FlexRankModel.from_pretrained(save_dir)

    assert loaded.physical_size_ratio == params[1] / params[0]
    assert _od_layer_state_max_dims(loaded) == {
        name: layer.inner_dim
        for name, layer in loaded.base_model.named_modules()
        if isinstance(layer, ODLayer)
    }


def test_reduce_size_can_deploy_with_svd_storage(tmp_path):
    """SVD deployment physically prunes without converting ODLinear layers to GAR."""
    save_dir = tmp_path / "flexrank-gpt2-pruned-sliced"
    config = transformers.GPT2Config(
        vocab_size=128,
        n_positions=16,
        n_embd=32,
        n_layer=1,
        n_head=4,
    )
    model = FlexRankModel.from_config(
        config,
        base_auto_class="AutoModelForCausalLM",
        exclude_layers=["lm_head"],
    )

    od_layer_names = [
        name for name, layer in model.base_model.named_modules() if isinstance(layer, ODLayer)
    ]
    od_layers = get_od_layers(model.base_model)
    profiles = [
        [max(1, int(layer.max_inner_dim * ratio)) for layer in od_layers] for ratio in (1.0, 0.5)
    ]
    params = [inner_dims_profile_to_params(profile, od_layers) for profile in profiles]
    model.profiles_data = ProfilesData(
        layers_name=od_layer_names,
        profiles=profiles,
        params=params,
        metrics=[{} for _ in profiles],
    )

    model.reduce_size(size_ratio=0.6, deploy_mode=DeployMode.SVD)
    model.save_pretrained(save_dir)

    saved_config = json.loads((save_dir / "config.json").read_text())
    assert "decomposed_layers" not in saved_config
    assert all(
        state["inner_dim"] == state["max_inner_dim"]
        for state in saved_config["od_layer_state"].values()
    )

    loaded = FlexRankModel.from_pretrained(save_dir)
    loaded_od_layers = {
        name: layer
        for name, layer in loaded.base_model.named_modules()
        if isinstance(layer, ODLayer)
    }

    assert loaded.physical_size_ratio == params[1] / params[0]
    assert all(layer.impl is ODImpl.DENSE for layer in loaded_od_layers.values())
    assert [loaded_od_layers[name].max_inner_dim for name in od_layer_names] == profiles[1]


def test_reduce_size_no_deploy_keeps_physical_size_ratio():
    """Virtual pruning should not physically prune stored tensors."""
    config = transformers.GPT2Config(
        vocab_size=128,
        n_positions=16,
        n_embd=32,
        n_layer=1,
        n_head=4,
    )
    model = FlexRankModel.from_config(
        config,
        base_auto_class="AutoModelForCausalLM",
        exclude_layers=["lm_head"],
    )

    od_layers = get_od_layers(model.base_model)
    profiles = [
        [max(1, int(layer.max_inner_dim * ratio)) for layer in od_layers] for ratio in (1.0, 0.5)
    ]
    params = [inner_dims_profile_to_params(profile, od_layers) for profile in profiles]
    model.profiles_data = ProfilesData(
        layers_name=[
            name for name, layer in model.base_model.named_modules() if isinstance(layer, ODLayer)
        ],
        profiles=profiles,
        params=params,
        metrics=[{} for _ in profiles],
    )

    model.reduce_size(size_ratio=0.6, deploy_mode=DeployMode.NO)

    assert model.virtual_size_ratio == params[1] / params[0]
    assert model.physical_size_ratio == 1.0
    assert model.config.virtual_size_ratio == 1.0
    assert model.active_profile == tuple(profiles[1])
    assert _od_layer_state_max_dims(model) == {
        name: layer.max_inner_dim
        for name, layer in model.base_model.named_modules()
        if isinstance(layer, ODLayer)
    }


def test_flexrank_model_from_pretrained_does_not_restore_virtual_profile(tmp_path):
    """Non-deployed active ranks are runtime-only and should not be saved."""
    save_dir = tmp_path / "flexrank-gpt2-virtual"
    config = transformers.GPT2Config(
        vocab_size=128,
        n_positions=16,
        n_embd=32,
        n_layer=1,
        n_head=4,
    )
    model = FlexRankModel.from_config(
        config,
        base_auto_class="AutoModelForCausalLM",
        exclude_layers=["lm_head"],
    )

    od_layer_names = [
        name for name, layer in model.base_model.named_modules() if isinstance(layer, ODLayer)
    ]
    od_layers = get_od_layers(model.base_model)
    profiles = [
        [max(1, int(layer.max_inner_dim * ratio)) for layer in od_layers] for ratio in (1.0, 0.5)
    ]
    params = [inner_dims_profile_to_params(profile, od_layers) for profile in profiles]
    model.profiles_data = ProfilesData(
        layers_name=od_layer_names,
        profiles=profiles,
        params=params,
        metrics=[{} for _ in profiles],
    )

    model.reduce_size(size_ratio=0.6, deploy_mode=DeployMode.NO)
    model.save_pretrained(save_dir)

    saved_config = json.loads((save_dir / "config.json").read_text())
    assert "decomposed_layers" not in saved_config
    assert saved_config["virtual_size_ratio"] == 1.0
    assert all(
        state["inner_dim"] == state["max_inner_dim"]
        for state in saved_config["od_layer_state"].values()
    )

    loaded = FlexRankModel.from_pretrained(save_dir)
    loaded_od_layers = {
        name: layer
        for name, layer in loaded.base_model.named_modules()
        if isinstance(layer, ODLayer)
    }

    assert loaded.active_profile is None
    assert loaded.virtual_size_ratio == 1.0
    assert loaded.physical_size_ratio == 1.0
    assert [loaded_od_layers[name].inner_dim for name in od_layer_names] == profiles[0]
    assert [loaded_od_layers[name].max_inner_dim for name in od_layer_names] == profiles[0]


def test_from_pretrained_materializes_meta_rank_masks(tmp_path):
    """Derived non-persistent rank masks should be usable after HF meta loading."""
    save_dir = tmp_path / "flexrank-gpt2-meta-mask"
    config = transformers.GPT2Config(
        vocab_size=128,
        n_positions=16,
        n_embd=32,
        n_layer=1,
        n_head=4,
    )
    model = FlexRankModel.from_config(
        config,
        base_auto_class="AutoModelForCausalLM",
        exclude_layers=["lm_head"],
    )
    model.save_pretrained(save_dir)

    loaded = FlexRankModel.from_pretrained(save_dir, low_cpu_mem_usage=True)
    rank_masks = [module._rank_mask for module in loaded.modules() if hasattr(module, "_rank_mask")]

    assert rank_masks
    assert all(not mask.is_meta for mask in rank_masks)
    loaded.to("cpu")
