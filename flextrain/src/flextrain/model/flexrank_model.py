"""
FlexRankModel: A HuggingFace-compatible wrapper for models with FlexRank decomposed layers.

Usage:
    # Wrap an existing decomposed model
    flex_model = FlexRankModel.from_model(model, training_args, decomposed_layer_names=[...])

    # Load from checkpoint
    flex_model = FlexRankModel.from_pretrained("./saved_model")

    # Create with random init (all layers decomposed)
    flex_model = FlexRankModel.from_config(config, base_model_type="gpt2")
"""

import functools
import inspect
import shutil
import types
from pathlib import Path
from typing import Any, Optional

import torch
import transformers
from torch import nn
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

try:
    from flexrank.layers import (
        build_decomp_like,
        get_decomposition_layers,
        replace_layer_by_name,
    )
    from flexrank.profiles import ProfilesData
    from flexrank.samplers.base_sampler import BaseSampler, DeployMode
    from flexrank.utils import train_only_specific_layers
except ImportError as exc:
    raise ImportError(
        "FlexRankModel checkpoints require the FlexRank runtime package. "
        "Install it with `pip install flexrank`, or install the `flexrank` "
        "package from the repository that published this checkpoint."
    ) from exc

__all__ = ["FlexRankConfig", "FlexRankModel"]

# Module path for auto_map (used in save_pretrained)
_MODULE_PATH = "flextrain.model.flexrank_model"
_HUB_REQUIREMENTS = ("flexrank>=0.0.1",)

# Infer Auto class from model class name suffix
_AUTO_CLASS_PATTERNS = (
    ("ForCausalLM", "AutoModelForCausalLM"),
    ("LMHeadModel", "AutoModelForCausalLM"),  # GPT2LMHeadModel, etc.
    ("ForMaskedLM", "AutoModelForMaskedLM"),
    ("ForSeq2SeqLM", "AutoModelForSeq2SeqLM"),
    ("ForSequenceClassification", "AutoModelForSequenceClassification"),
    ("ForTokenClassification", "AutoModelForTokenClassification"),
    ("ForQuestionAnswering", "AutoModelForQuestionAnswering"),
    ("ForImageClassification", "AutoModelForImageClassification"),
    ("ForObjectDetection", "AutoModelForObjectDetection"),
)


def _infer_auto_class(model: nn.Module) -> str:
    """Infer Auto class from model's _auto_class attr or class name suffix."""
    if auto_cls := getattr(model, "_auto_class", None):
        return auto_cls
    cls_name = type(model).__name__
    for suffix, auto_class in _AUTO_CLASS_PATTERNS:
        if suffix in cls_name:
            return auto_class
    return "AutoModel"


def _get_inner_dim(layer: nn.Module) -> int:
    """Get inner_dim from a FlexRank decomposed layer (property or method)."""
    inner_dim = getattr(layer, "inner_dim", None)
    assert inner_dim is not None, f"{type(layer).__name__} has no inner_dim"
    return inner_dim() if callable(inner_dim) else inner_dim


def _get_max_inner_dim(layer: nn.Module) -> int:
    """Get max_inner_dim from a FlexRank decomposed layer."""
    max_inner_dim = getattr(layer, "max_inner_dim", None)
    assert max_inner_dim is not None, f"{type(layer).__name__} has no max_inner_dim"
    return max_inner_dim() if callable(max_inner_dim) else max_inner_dim


def _get_decomposed_layers(
    model: nn.Module, decomposed_layer_names: list[str]
) -> dict[str, nn.Module]:
    return {name: model.get_submodule(name) for name in decomposed_layer_names}


def _get_decomposed_layers_state(
    decomposed_layers: dict[str, nn.Module],
) -> dict[str, dict]:
    return {name: layer.export_flexrank_state() for name, layer in decomposed_layers.items()}


def _profiles_data_from_config(data: Optional[dict]) -> Optional["ProfilesData"]:
    """Rebuild profile data from compact deployment config."""
    if not data:
        return None
    payload = dict(data)
    payload.setdefault("metrics", [])
    return ProfilesData(**payload)


def _profiles_data_to_config(profiles_data: Optional["ProfilesData"]) -> Optional[dict]:
    """Serialize profile data through its public deployment export interface."""
    if profiles_data is None:
        return None
    return profiles_data.to_export_dict()


def _compact_profiles_data_config(data: Optional[dict]) -> Optional[dict]:
    """Drop non-deployment profile metadata from a serialized config payload."""
    if not data:
        return None
    return {
        "layers_name": data["layers_name"],
        "profiles": data["profiles"],
        "params": data["params"],
    }


def _build_base_config(base_model_type: str, base_model_config: dict) -> PretrainedConfig:
    """Build a HF base config from an exact serialized config payload."""
    config_kwargs = dict(base_model_config)
    serialized_model_type = config_kwargs.pop("model_type", base_model_type)
    if serialized_model_type != base_model_type:
        raise ValueError(
            f"base_model_type={base_model_type!r} does not match "
            f"base_model_config.model_type={serialized_model_type!r}"
        )
    return AutoConfig.for_model(base_model_type, **config_kwargs)


def make_model_contiguous(model: nn.Module) -> None:
    """Make model parameters contiguous in memory."""
    for param in model.parameters():
        param.data = param.data.contiguous()


def _conv1d_to_linear(
    layer: transformers.models.gpt2.modeling_gpt2.Conv1D,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Linear:
    """Create an equivalent Linear layer from a GPT-2 Conv1D layer."""
    device = device or layer.weight.device
    dtype = dtype or layer.weight.dtype

    linear = nn.Linear(
        layer.nx,
        layer.nf,
        bias=layer.bias is not None,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(layer.weight.T.contiguous())
        if layer.bias is not None:
            linear.bias.copy_(layer.bias.contiguous())
    return linear


def replace_conv1d_with_linear(
    model: nn.Module,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Recursively replace GPT-2 Conv1D layers with equivalent Linear layers."""
    for name, module in model.named_children():
        if isinstance(module, transformers.models.gpt2.modeling_gpt2.Conv1D):
            new_module = _conv1d_to_linear(module, device=device, dtype=dtype)
            make_model_contiguous(new_module)
            setattr(model, name, new_module)
        else:
            replace_conv1d_with_linear(module, device=device, dtype=dtype)

    return model


def _ensure_hub_requirements(save_path: Path) -> None:
    """Ensure Hub model repositories advertise required runtime packages."""
    requirements_path = save_path / "requirements.txt"
    existing = (
        requirements_path.read_text(encoding="utf-8").splitlines()
        if requirements_path.exists()
        else []
    )
    normalized = {
        line.strip().split("=", 1)[0].split(">", 1)[0].split("<", 1)[0]
        for line in existing
        if line.strip() and not line.lstrip().startswith("#")
    }
    additions = [
        requirement
        for requirement in _HUB_REQUIREMENTS
        if requirement.split("=", 1)[0].split(">", 1)[0].split("<", 1)[0] not in normalized
    ]
    if not additions:
        return

    requirements_path.write_text(
        "\n".join([*existing, *additions]).strip() + "\n",
        encoding="utf-8",
    )


def _make_contiguous_and_set_trainable(
    model: nn.Module,
    trainable_layers: dict[str, nn.Module],
    freeze_non_decomposed: bool,
) -> None:
    make_model_contiguous(model)

    if freeze_non_decomposed:
        train_only_specific_layers(model, trainable_layers.items())


def _resolve_freeze_non_decomposed(
    config: "FlexRankConfig",
    freeze_non_decomposed: Optional[bool],
) -> bool:
    """Resolve the freeze flag from the explicit argument or serialized config."""
    if freeze_non_decomposed is not None:
        return freeze_non_decomposed

    decomposition_config = config.decomposition_config or {}
    return decomposition_config.get("freeze_non_decomposed", False)


def _bind_forward_to_wrapped_model(model: PreTrainedModel) -> callable:
    """Create an instance-bound forward wrapper that preserves the wrapped signature."""
    wrapped_signature = inspect.signature(model.forward)

    @functools.wraps(model.forward)
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    forward.__signature__ = wrapped_signature
    return forward


def _init_random_flexmodel_as(
    base_model: PreTrainedModel,
    exclude_layers: list[str],
    freeze_non_decomposed: bool = False,
) -> dict[str, int]:
    dcmp_layers_dim = {}
    dcmp_layers = {}
    for name, layer in get_decomposition_layers(base_model, exclude_layers):
        decomp_layer = build_decomp_like(layer)
        replace_layer_by_name(base_model, name, decomp_layer)
        dcmp_layers_dim[name] = _get_inner_dim(decomp_layer)
        dcmp_layers[name] = decomp_layer

    _make_contiguous_and_set_trainable(base_model, dcmp_layers, freeze_non_decomposed)

    return dcmp_layers_dim


class FlexRankConfig(PretrainedConfig):
    """Config for FlexRank models, storing decomposition metadata and training config."""

    model_type = "flexrank"

    _METADATA_KEYS = (
        "architecture_meta",
        "dataset_config",
        "decomposition_config",
        "profile_algo_config",
        "sampler_config",
        "training_config",
        "model_config",
        "profiles_data",
        "od_layer_state",
    )

    def __init__(
        self,
        base_model_type: str = "",
        base_auto_class: str = "AutoModel",
        base_model_config: Optional[dict] = None,
        architecture_meta: Optional[dict] = None,
        dataset_config: Optional[dict] = None,
        decomposition_config: Optional[dict] = None,
        profile_algo_config: Optional[dict] = None,
        sampler_config: Optional[dict] = None,
        training_config: Optional[dict] = None,
        model_config: Optional[dict] = None,
        profiles_data: Optional[dict] = None,
        od_layer_state: Optional[dict] = None,
        physical_size_ratio: float = 1.0,
        virtual_size_ratio: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_type = base_model_type
        self.base_auto_class = base_auto_class
        self.base_model_config = base_model_config or {}
        self._metadata = {
            "architecture_meta": architecture_meta or {},
            "dataset_config": dataset_config,
            "decomposition_config": decomposition_config,
            "profile_algo_config": profile_algo_config,
            "sampler_config": sampler_config,
            "training_config": training_config,
            "model_config": model_config,
            "profiles_data": profiles_data,
            "od_layer_state": od_layer_state,
        }
        self._metadata["architecture_meta"] = self._metadata["architecture_meta"] or {}
        self._metadata["profiles_data"] = _compact_profiles_data_config(
            self._metadata["profiles_data"]
        )
        self._size_ratios = {
            "physical": physical_size_ratio,
            "virtual": virtual_size_ratio,
        }

    def to_dict(self) -> dict:
        """Serialize base config fields together with FlexRank metadata."""
        output = super().to_dict()
        output.pop("_metadata", None)
        output.pop("_size_ratios", None)
        for key in self._METADATA_KEYS:
            value = self._metadata[key]
            if key == "profiles_data":
                value = _compact_profiles_data_config(value)
            output[key] = value
        output["physical_size_ratio"] = self.physical_size_ratio
        output["virtual_size_ratio"] = self.virtual_size_ratio
        return output

    @property
    def architecture_meta(self) -> dict:
        """Serialized metadata about how the FlexRank model was created."""
        return self._metadata["architecture_meta"]

    @architecture_meta.setter
    def architecture_meta(self, value: Optional[dict]) -> None:
        self._metadata["architecture_meta"] = value or {}

    @property
    def dataset_config(self) -> Optional[dict]:
        """Exported dataset configuration used to build the checkpoint."""
        return self._metadata["dataset_config"]

    @dataset_config.setter
    def dataset_config(self, value: Optional[dict]) -> None:
        self._metadata["dataset_config"] = value

    @property
    def decomposition_config(self) -> Optional[dict]:
        """Exported decomposition settings stored with the checkpoint."""
        return self._metadata["decomposition_config"]

    @decomposition_config.setter
    def decomposition_config(self, value: Optional[dict]) -> None:
        self._metadata["decomposition_config"] = value

    @property
    def profile_algo_config(self) -> Optional[dict]:
        """Exported profile-search configuration."""
        return self._metadata["profile_algo_config"]

    @profile_algo_config.setter
    def profile_algo_config(self, value: Optional[dict]) -> None:
        self._metadata["profile_algo_config"] = value

    @property
    def sampler_config(self) -> Optional[dict]:
        """Exported sampler configuration."""
        return self._metadata["sampler_config"]

    @sampler_config.setter
    def sampler_config(self, value: Optional[dict]) -> None:
        self._metadata["sampler_config"] = value

    @property
    def training_config(self) -> Optional[dict]:
        """Exported training configuration."""
        return self._metadata["training_config"]

    @training_config.setter
    def training_config(self, value: Optional[dict]) -> None:
        self._metadata["training_config"] = value

    @property
    def model_config(self) -> Optional[dict]:
        """Exported base-model loading configuration."""
        return self._metadata["model_config"]

    @model_config.setter
    def model_config(self, value: Optional[dict]) -> None:
        self._metadata["model_config"] = value

    @property
    def profiles_data(self) -> Optional[dict]:
        """Serialized profile statistics stored in the checkpoint config."""
        return self._metadata["profiles_data"]

    @profiles_data.setter
    def profiles_data(self, value: Optional[dict]) -> None:
        self._metadata["profiles_data"] = _compact_profiles_data_config(value)

    @property
    def od_layer_state(self) -> Optional[dict]:
        """Serialized per-OD-layer metadata needed to reconstruct storage state."""
        return self._metadata["od_layer_state"]

    @od_layer_state.setter
    def od_layer_state(self, value: Optional[dict]) -> None:
        self._metadata["od_layer_state"] = value

    @property
    def physical_size_ratio(self) -> float:
        """Ratio of deployed parameters relative to the full FlexRank model."""
        return self._size_ratios["physical"]

    @physical_size_ratio.setter
    def physical_size_ratio(self, value: float) -> None:
        self._size_ratios["physical"] = value

    @property
    def virtual_size_ratio(self) -> float:
        """Target sampling ratio currently requested for inference."""
        return self._size_ratios["virtual"]

    @virtual_size_ratio.setter
    def virtual_size_ratio(self, value: float) -> None:
        self._size_ratios["virtual"] = value

    @property
    def exclude_layers(self) -> list[str]:
        """Get exclude_layers_names from decomposition_config."""
        if self.decomposition_config:
            return self.decomposition_config.get("exclude_layers_names", [])
        return []


class FlexRankModel(PreTrainedModel):
    """HuggingFace-compatible wrapper for models with FlexRank decomposed layers."""

    config_class = FlexRankConfig
    base_model_prefix = "_wrapped_model"
    config: FlexRankConfig  # Type hint for IDE support

    def __init__(
        self,
        config: FlexRankConfig,
        base_model: Optional[PreTrainedModel] = None,
        profiles_data: Optional["ProfilesData"] = None,
    ):
        super().__init__(config)

        # Build from config if no base_model provided (for from_pretrained)
        if base_model is None:
            if not config.od_layer_state:
                raise ValueError("od_layer_state required to build model")
            base_model = self._build_base_model(config)

        self._wrapped_model = base_model
        if hasattr(self._wrapped_model, "config"):
            self._wrapped_model.config.use_cache = False
        self._decomposed_layer_names = frozenset(config.od_layer_state or {})
        self._active_profile: Optional[tuple[int, ...]] = None
        self._virtual_size_ratio = config.virtual_size_ratio

        # Expose tied weights for HF Trainer checkpoint loading (with wrapper prefix)
        base_tied = getattr(base_model, "_tied_weights_keys", []) or []
        self._tied_weights_keys = [f"{self.base_model_prefix}.{k}" for k in base_tied]
        self._keys_to_ignore_on_save = self._tied_weights_keys

        # Reconstruct ProfilesData from config if not provided
        if profiles_data is None:
            profiles_data = _profiles_data_from_config(config.profiles_data)
        self._profiles_data = profiles_data
        if profiles_data is not None and not self.config.profiles_data:
            self.config.profiles_data = _profiles_data_to_config(profiles_data)
        self.forward = types.MethodType(_bind_forward_to_wrapped_model(base_model), self)

    def __getattr__(self, name: str):
        """Delegate to wrapped model for methods like generate()."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        wrapped = object.__getattribute__(self, "_modules").get("_wrapped_model")
        if wrapped is not None:
            return getattr(wrapped, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @staticmethod
    def _fix_state_dict_key_on_load(key: str) -> tuple[str, bool]:
        """Preserve FlexRank factor names while keeping HF's standard load renames."""
        if key.endswith("weight_v") and "._wrapped_model." in f".{key}":
            return key, False
        return PreTrainedModel._fix_state_dict_key_on_load(key)

    def _sync_od_layer_state_config(self) -> None:
        """Keep serialized physical OD storage state aligned with module shapes."""
        dcmp_layers = _get_decomposed_layers(
            self._wrapped_model, list(self._decomposed_layer_names)
        )
        self.config.od_layer_state = _get_decomposed_layers_state(dcmp_layers)

    def state_dict(self, *args, **kwargs):
        """Return state without OD extra-state entries for HF/FSDP compatibility.

        OD metadata is serialized in `config.od_layer_state`. Keeping PyTorch
        `_extra_state` entries in this wrapper state dict trips FSDP2's internal
        sharded-state reload because those entries are plain tensors rather than
        DTensors.
        """
        state = super().state_dict(*args, **kwargs)
        for key in [key for key in state if key.endswith("._extra_state")]:
            del state[key]
        return state

    def materialize_runtime_buffers(self) -> "FlexRankModel":
        """Materialize derived non-persistent buffers after checkpoint loading."""
        for module in self.modules():
            if module is self:
                continue
            materialize = getattr(module, "materialize_runtime_buffers", None)
            if materialize is not None:
                materialize()
        return self

    # -------------------------------------------------------------------------
    # Model building
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_base_model(config: FlexRankConfig) -> PreTrainedModel:
        """Build base model with decomposed layers from config."""
        if not config.base_model_type:
            raise ValueError("base_model_type required")

        base_config = _build_base_config(
            config.base_model_type,
            config.base_model_config,
        )
        auto_cls = getattr(transformers, config.base_auto_class, transformers.AutoModel)
        base_model = auto_cls.from_config(base_config)
        replace_conv1d_with_linear(base_model)

        if not config.od_layer_state:
            raise ValueError("od_layer_state required to build model")
        for layer_name, layer_state in config.od_layer_state.items():
            expected_dim = int(layer_state["max_inner_dim"])
            layer = base_model.get_submodule(layer_name)
            decomp_layer = build_decomp_like(layer)
            decomp_layer.set_extra_state(layer_state)
            actual_dim = _get_max_inner_dim(decomp_layer)
            if actual_dim != expected_dim:
                raise ValueError(f"{layer_name}: expected {expected_dim}, got {actual_dim}")
            replace_layer_by_name(base_model, layer_name, decomp_layer)

        _make_contiguous_and_set_trainable(
            base_model,
            _get_decomposed_layers(base_model, list(config.od_layer_state)),
            _resolve_freeze_non_decomposed(config, None),
        )
        return base_model

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "FlexRankModel":
        """Load a FlexRank checkpoint and rebuild derived runtime buffers."""
        model = super().from_pretrained(*args, **kwargs)
        return model.materialize_runtime_buffers()

    @classmethod
    def from_model(
        cls,
        model: PreTrainedModel,
        training_args: Any,
        **kwargs,
    ) -> "FlexRankModel":
        """Wrap an existing model. Auto-detects base_model_type from model.config."""
        base_model_type = kwargs.pop("base_model_type", None)
        decomposed_layer_names = kwargs.pop("decomposed_layer_names", None)
        profiles_data = kwargs.pop("profiles_data", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        base_model_type = base_model_type or getattr(
            getattr(model, "config", None), "model_type", "custom"
        )
        decomp_args = training_args.decomposition
        model_config = model.config.to_dict() if hasattr(model, "config") else {}

        dcmp_layers = _get_decomposed_layers(model, decomposed_layer_names or [])
        dcmp_layers_state = _get_decomposed_layers_state(dcmp_layers)

        config = FlexRankConfig(
            base_model_type=base_model_type,
            base_auto_class=_infer_auto_class(model),
            base_model_config=model_config,
            dataset_config=training_args.dataset.to_export_dict(),
            decomposition_config=decomp_args.to_export_dict(),
            profile_algo_config=training_args.profile_algo.to_export_dict(),
            sampler_config=training_args.sampler.to_export_dict(),
            training_config=training_args.train.to_export_dict(),
            model_config=training_args.model.to_export_dict(),
            profiles_data=_profiles_data_to_config(profiles_data),
            od_layer_state=dcmp_layers_state,
        )

        _make_contiguous_and_set_trainable(model, dcmp_layers, decomp_args.freeze_non_decomposed)

        return cls(config, base_model=model, profiles_data=profiles_data)

    @classmethod
    def from_config(
        cls,
        config,
        **kwargs,
    ) -> "FlexRankModel":
        """Create FlexRankModel with random init, decomposing all (non-excluded) layers."""
        base_model_type = kwargs.pop("base_model_type", None)
        base_auto_class = kwargs.pop("base_auto_class", None)
        exclude_layers = kwargs.pop("exclude_layers", None)
        freeze_non_decomposed = kwargs.pop("freeze_non_decomposed", None)
        model_kwargs = kwargs

        if isinstance(config, str):
            config = AutoConfig.from_pretrained(config)

        # If FlexRankConfig with OD layer state, rebuild exact architecture
        if isinstance(config, FlexRankConfig) and config.od_layer_state:
            return cls(config)

        # Infer parameters from config (with fallbacks)
        is_flexrank = isinstance(config, FlexRankConfig)
        base_model_type = (
            base_model_type
            or (config.base_model_type if is_flexrank else None)
            or (config.model_type if config.model_type != "flexrank" else None)
        )
        if not base_model_type:
            raise ValueError("Cannot determine base_model_type from config")

        base_auto_class = base_auto_class or (
            config.base_auto_class if is_flexrank else "AutoModel"
        )
        exclude_layers = exclude_layers or (config.exclude_layers if is_flexrank else [])
        if is_flexrank:
            freeze_non_decomposed = _resolve_freeze_non_decomposed(config, freeze_non_decomposed)
        else:
            freeze_non_decomposed = bool(freeze_non_decomposed)

        # Build base model
        base_config = (
            _build_base_config(base_model_type, config.base_model_config) if is_flexrank else config
        )
        auto_cls = getattr(transformers, base_auto_class, transformers.AutoModel)
        base_model = auto_cls.from_config(base_config, **model_kwargs)

        replace_conv1d_with_linear(base_model)

        # Replace layers with decomposed versions
        decomposed_layer_dims = _init_random_flexmodel_as(
            base_model,
            exclude_layers,
            freeze_non_decomposed,
        )

        flex_config = FlexRankConfig(
            base_model_type=base_model_type,
            base_auto_class=base_auto_class,
            base_model_config=base_config.to_dict(),
            od_layer_state=_get_decomposed_layers_state(
                _get_decomposed_layers(base_model, list(decomposed_layer_dims))
            ),
            decomposition_config={"exclude_layers_names": exclude_layers},
            architecture_meta={"random_init": True},
        )

        return cls(flex_config, base_model=base_model)

    # -------------------------------------------------------------------------
    # Saving
    # -------------------------------------------------------------------------

    def save_pretrained(
        self,
        save_directory: str,
        copy_source: bool = True,
        write_requirements: bool = True,
        **kwargs,
    ):
        """Save FlexRankModel to directory.

        Args:
            save_directory: Directory to save to.
            copy_source: If True, copies this file so model loads in fresh environments.
            write_requirements: If True, writes Hub runtime dependencies.
            **kwargs: Passed to HF's save_pretrained.
        """
        save_path = Path(save_directory)

        if copy_source or write_requirements:
            save_path.mkdir(parents=True, exist_ok=True)

        # Copy source file for standalone loading
        if copy_source:
            src = Path(__file__)
            dst = save_path / src.name
            if not dst.exists():
                shutil.copy(src, dst)
            module_path = src.stem
        else:
            module_path = _MODULE_PATH

        if write_requirements:
            _ensure_hub_requirements(save_path)

        # Set auto_map before HF saves config
        self.config.auto_map = {
            "AutoConfig": f"{module_path}.FlexRankConfig",
            self.config.base_auto_class: f"{module_path}.FlexRankModel",
        }

        state_dict = kwargs.pop("state_dict", None)
        if state_dict is None:
            state_dict = self.state_dict()
        state_dict = {
            key: value for key, value in state_dict.items() if not key.endswith("._extra_state")
        }

        # Delegate to HF (handles weights, config, sharding, etc.)
        super().save_pretrained(save_directory, state_dict=state_dict, **kwargs)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def base_model(self) -> PreTrainedModel:
        """Wrapped model (alias for _wrapped_model)."""
        return self._wrapped_model

    @property
    def decomposed_layer_names(self) -> frozenset[str]:
        """Names of the decomposed layers tracked by this wrapper."""
        return self._decomposed_layer_names

    @property
    def profiles_data(self) -> Optional["ProfilesData"]:
        """Profiles metadata currently attached to the model."""
        return self._profiles_data

    @profiles_data.setter
    def profiles_data(self, value: Optional["ProfilesData"]):
        """Keep the in-memory profiles object and serialized config in sync."""
        self._profiles_data = value
        self.config.profiles_data = _profiles_data_to_config(value)

    @property
    def physical_size_ratio(self) -> float:
        """Current physical size ratio after physical deployment. Stored in config."""
        return self.config.physical_size_ratio

    @property
    def virtual_size_ratio(self) -> float:
        """Current virtual size ratio for this runtime model instance."""
        return self._virtual_size_ratio

    @property
    def active_profile(self) -> Optional[tuple[int, ...]]:
        """Runtime-only active profile currently applied to OD layers."""
        return self._active_profile

    # -------------------------------------------------------------------------
    # Inference utilities
    # -------------------------------------------------------------------------

    def reduce_size(
        self,
        *,
        size_ratio: Optional[float] = None,
        compression_rate: Optional[float] = None,
        deploy_mode: DeployMode = DeployMode.NO,
    ) -> None:
        """
        Reduce model size to a target ratio or compression rate.
        This method prunes the model to achieve a specified size reduction, either by
        specifying a target size ratio or a compression rate. The method uses profiling
        data to determine which layers to prune and applies the pruning configuration
        to the wrapped model.
        Args:
            size_ratio (Optional[float]): Target size as a ratio of the original model size.
                Must be between 0 and 1. If None, compression_rate must be specified.
                Defaults to None.
            compression_rate (Optional[float]): Compression rate (0 to 1) where the resulting
                size_ratio = 1.0 - compression_rate. If None, size_ratio must be specified.
                Defaults to None.
            deploy_mode (DeployMode): Whether and how to physically deploy the selected
                profile. Use ``DeployMode.NO`` ("NO") to only change active ranks
                virtually, ``DeployMode.SVD`` ("SVD") to physically prune stored SVD
                factors, or ``DeployMode.GAR`` ("GAR") to physically prune and use
                GAR storage where supported. Layers without GAR support fall back to
                SVD storage.
        Raises:
            ValueError: If both size_ratio and compression_rate are None.
            AssertionError: If profiles_data is not available or empty.
        Returns:
            None
        Example:
            # Reduce model to 50% of original size
            model.reduce_size(size_ratio=0.5)
            # Deploy model with 30% compression using GAR where available
            model.reduce_size(compression_rate=0.3, deploy_mode=DeployMode.GAR)
        """
        if size_ratio is None and compression_rate is None:
            raise ValueError("size_ratio or compression_rate required")

        size_ratio = size_ratio if size_ratio is not None else 1.0 - compression_rate
        assert 0 < size_ratio <= 1, "size_ratio must be between 0 (excluded) and 1 (included)"
        assert self.profiles_data, "profiles_data required for reduce_size"

        target_params = int(self.profiles_data.max_params * size_ratio)
        assert self.physical_size_ratio >= size_ratio, (
            "Cannot increase model size: "
            f"current size_ratio={self.physical_size_ratio:.3f}x "
            f"(~{int(self.profiles_data.max_params * self.physical_size_ratio):,} params) "
            f"requested size_ratio={size_ratio:.3f}x "
            f"(~{target_params:,} params). "
            "Use a smaller size_ratio or larger compression_rate."
        )

        profile, selected_params = self.profiles_data.get_profile_for_params(target_params)
        selected_size_ratio = selected_params / self.profiles_data.max_params
        BaseSampler(self._wrapped_model).set_p_for_layers(profile, deploy_mode=deploy_mode)
        self._active_profile = tuple(profile)
        self._virtual_size_ratio = selected_size_ratio

        if deploy_mode is not DeployMode.NO:
            self._sync_od_layer_state_config()
            self.config.physical_size_ratio = selected_size_ratio
            self.config.virtual_size_ratio = selected_size_ratio


# Register for AutoModel loading
FlexRankConfig.register_for_auto_class()
FlexRankModel.register_for_auto_class("AutoModel")
