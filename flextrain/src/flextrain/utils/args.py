"""Argument parsing logic"""

import datetime
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, cast

import hydra
import omegaconf
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers import TrainingArguments
from transformers.utils import logging

from flexrank.types import SVDType

log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = {"passive": -1, **log_levels}


__all__ = [
    "parse_structured_config",
    "Config",
    "SamplerArguments",
    "NLPModelArguments",
    "VisionModelArguments",
    "NLPDataArguments",
    "VisionDataArguments",
    "WandbArguments",
    "DecompositionArguments",
    "LMEvalArguments",
    "TrainArguments",
    "DistillTrainArguments",
    "SerializableMixin",
    "SVDType",
]

NOW = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')}"

_ConfigT = TypeVar("_ConfigT")
_ReturnT = TypeVar("_ReturnT")


class SerializableMixin:
    """Mixin that serializes dataclass-based config objects."""

    # Override in subclasses to whitelist exportable fields
    _export_fields: list[str] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a JSON-serializable dictionary."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            result[f.name] = self.serialize_value(value)
        return result

    def to_export_dict(self) -> Dict[str, Any]:
        """Convert the whitelisted export fields to a JSON-serializable dictionary."""
        result = {}
        for f in fields(self):
            if f.name in self._export_fields:
                value = getattr(self, f.name)
                result[f.name] = self.serialize_value(value)
        return result

    @staticmethod
    def serialize_value(value: Any) -> Any:
        """Recursively serialize a value to JSON-compatible types."""
        serialized = value
        if value is None:
            serialized = None
        elif isinstance(value, SerializableMixin):
            serialized = value.to_dict()
        elif hasattr(value, "to_dict") and callable(value.to_dict):
            serialized = value.to_dict()
        elif isinstance(value, torch.dtype):
            serialized = str(value)
        elif isinstance(value, (list, tuple)):
            serialized = [SerializableMixin.serialize_value(v) for v in value]
        elif isinstance(value, dict):
            serialized = {k: SerializableMixin.serialize_value(v) for k, v in value.items()}
        elif isinstance(value, StrEnum):
            serialized = value.value
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            # Handle other dataclasses that don't inherit from SerializableMixin.
            try:
                serialized = asdict(value)
            except TypeError:
                serialized = str(value)
        return serialized

    _serialize_value = serialize_value


@dataclass
class WandbArguments(SerializableMixin):
    """Configuration arguments for Weights & Biases integration."""

    project: str = field(metadata={"help": "The name of the project where to log the new run onto"})

    entity: str = field(
        metadata={"help": "An entity is a username or team name where to log the new run onto"}
    )

    name: str = field(default=NOW, metadata={"help": "The name to assign to the new run"})

    api_key_file: Optional[str] = field(
        default=None,
        metadata={
            "help": 'Path to a JSON file containing {"WANDB_API_KEY": "..."}. "'
            "Used only at runtime and never saved."
        },
    )

    id: Optional[str] = field(
        default=None, metadata={"help": "A unique ID for the run, used for resuming"}
    )

    resume: str = field(
        default="allow",
        metadata={
            "help": "Controls the behavior when resuming a run with the specified `id`.",
            "options": ["allow", "never", "must", "auto"],
        },
    )

    dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The absolute path to the directory where experiment logs and "
            "metadata files are stored. If not specified, this defaults "
            "to the `./wandb` directory. "
        },
    )

    mode: str = field(
        default="online",
        metadata={
            "help": "Specifies how run data is managed",
            "options": ["online", "offline", "disabled"],
        },
    )


@dataclass
class BaseModelArguments(SerializableMixin):
    """Base arguments for loading and initializing Hugging Face models."""

    model_name_or_path: str = field(
        default=omegaconf.MISSING,
        metadata={
            "help": "A string, the *model id* of a pretrained model hosted inside a model repo on "
            "huggingface.co or A path to a *directory* containing model weights saved"
            "using [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`."
        },
    )

    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )

    use_auth_token: bool = field(
        default=True,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."},
    )

    load_pretrained_model: bool = field(
        default=True, metadata={"help": "Whether to load pretrained weights from HF."}
    )

    _torch_dtype: str = field(
        default=omegaconf.MISSING,
        metadata={"help": "The torch dtype to pass to AutoModelForCausalLM."},
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Cache directory for AutoModelForCausalLM"}
    )

    @property
    def torch_dtype(self) -> torch.dtype:
        """Converts the private torch_dtype string to a torch.dtype object"""
        return getattr(torch, self._torch_dtype.rsplit(".", maxsplit=1)[-1])


@dataclass
class NLPModelArguments(BaseModelArguments):
    """NLP model arguments configuration."""

    _export_fields = ["model_name_or_path"]


@dataclass
class VisionModelArguments(BaseModelArguments):
    """Vision model arguments configuration."""

    _export_fields = ["model_name_or_path", "num_labels"]

    num_labels: int = field(
        default=omegaconf.MISSING,
        metadata={
            "help": "Number of labels of the dataset used, needed to reinitialize the head "
            "if necessary. It must match the corresponding argument in DatasetArguments. "
            "Valid only for image classification task"
        },
    )


@dataclass
class BaseDataArguments(SerializableMixin):
    """Base arguments for loading and preparing Hugging Face datasets."""

    path: str = field(default=omegaconf.MISSING, metadata={"help": "Which dataset to train on."})

    name: str = field(
        default=omegaconf.MISSING,
        metadata={"help": "The name of the split of the dataset to use."},
    )

    streaming: bool = field(
        default=True, metadata={"help": "Whether to use the dataset in streaming mode."}
    )

    cache_file_names: Optional[dict[str, str]] = field(
        default=None,
        metadata={
            "help": "Directory to read/write data, for each of the splits of the dataset."
            "Defaults to `None`."
        },
    )

    eval_ds_size: int = field(
        default=omegaconf.MISSING,
        metadata={
            "help": "Size of evaluation dataset, extracted from the training data"
            "if the a validation set is not available in the original dataset."
            "Used for evaluation during training"
        },
    )

    calib_ds_size: int = field(
        default=omegaconf.MISSING,
        metadata={
            "help": "Size of calibration dataset, extracted from the training data"
            "if the a validation set is not available in the original dataset."
        },
    )

    def __post_init__(self):
        assert self.eval_ds_size > 0, "Invalid dimension for validation dataset"
        assert self.calib_ds_size > 0, "Invalid dimension for calibration dataset"


@dataclass
class NLPDataArguments(BaseDataArguments):
    """Data arguments for NLP tasks."""

    _export_fields = [
        "path",
        "name",
        "eval_ds_size",
        "calib_ds_size",
        "sequence_length",
    ]

    sequence_length: int = field(
        default=omegaconf.MISSING, metadata={"help": "Sequence lenght for the model"}
    )


@dataclass
class VisionDataArguments(BaseDataArguments):
    """Data arguments for vision tasks."""

    _export_fields = [
        "path",
        "name",
        "eval_ds_size",
        "calib_ds_size",
        "num_labels",
        "resize_size",
        "crop_size",
    ]

    num_labels: int = field(
        default=omegaconf.MISSING,
        metadata={
            "help": "Number of labels of the dataset used,"
            "needed to reinitialize the head if necessary."
        },
    )

    resize_size: int = field(
        default=omegaconf.MISSING,
        metadata={"help": "Size to resize the input images to. "},
    )

    crop_size: int = field(
        default=omegaconf.MISSING,
        metadata={"help": "Size to crop the input images to. "},
    )

    image_col: str = field(
        default=omegaconf.MISSING,
        metadata={"help": "Name of the image field in the dataset. "},
    )

    label_col: str = field(
        default=omegaconf.MISSING,
        metadata={"help": "Name of the label field in the dataset. "},
    )


@dataclass
class DecompositionArguments(SerializableMixin):
    """Configuration arguments for model decomposition strategies."""

    _export_fields = ["svd_type", "min_p", "n_models", "freeze_non_decomposed"]

    svd_type: SVDType = field(
        default=SVDType.SVD,
        metadata={
            "help": "SVD decomposition mode",
            "options": list(SVDType),
        },
    )

    min_p: float = field(
        default=omegaconf.MISSING,
        metadata={"help": "Size of the smallest submodel, expressed as portion of the base model"},
    )

    n_models: int = field(
        default=omegaconf.MISSING, metadata={"help": "Number of submodels to train"}
    )

    freeze_non_decomposed: bool = field(
        default=omegaconf.MISSING,
        metadata={"help": "Whether to freeze or train layers that have not been decomposed"},
    )

    exclude_layers_names: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of strings, corresponding to (partial) layer_names"
            "to exlude from decomposition"
        },
    )

    gram_cache_device: str = field(
        default="cuda",
        metadata={
            "help": "Type of device to store the layers' Gram matrices in case of DataSVD",
            "options": ["cpu", "cuda"],
        },
    )

    def __post_init__(self):
        assert self.gram_cache_device in ["cpu", "cuda"], "Unknown device"


@dataclass
class ProfileSearchAlgoArguments(SerializableMixin):
    """Arguments for the profile searching algotithm"""

    _export_fields = ["classname", "algo_kwargs"]

    classname: str = field(
        default=omegaconf.MISSING,
        metadata={
            "help": "Classname of the algorithm to use for searching profiles",
            "options": [
                "DPSearchAlgo",
            ],
        },
    )

    algo_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Constructor arguments specific to algorithm classname"},
    )

    eager_pruning: bool = field(
        default=False, metadata={"help": "Whether or not to prune models after search."}
    )


@dataclass
class SamplerArguments(SerializableMixin):
    """Configuration arguments for sampler initialization."""

    _export_fields = ["classname", "sampler_kwargs"]

    classname: str = field(
        default=omegaconf.MISSING,
        metadata={
            "help": "Name of sampler class to instantiate",
            "options": [
                "AllLayerSampler",
                "SingleLayerSampler",
                "SingleLayerPowerTwoSampler",
                "PredefinedModelSampler",
            ],
        },
    )

    sampler_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Constructor arguments specific to sampler classname"},
    )


@dataclass
class LMEvalArguments(SerializableMixin):
    """Configuration arguments for lm_eval downstream evaluation."""

    _export_fields = [
        "enabled",
        "task_names",
        "batch_size",
        "num_fewshot",
        "eval_steps",
        "limit",
        "bootstrap_iters",
        "cache_requests",
        "log_samples",
        "suppress_output",
    ]

    enabled: bool = field(
        default=False,
        metadata={"help": "Whether to run downstream evaluation with lm_eval"},
    )

    task_names: list[str] = field(
        default_factory=list,
        metadata={"help": "List of lm_eval task names to evaluate"},
    )

    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for lm_eval"},
    )

    num_fewshot: int = field(
        default=0, metadata={"help": "Number of few-shot examples for lm_eval"}
    )

    eval_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Run lm_eval every N steps (None uses the trainer eval schedule)"},
    )

    limit: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Limit examples per lm_eval task. Values below 1 are interpreted "
                "by lm_eval as a fraction of each task."
            )
        },
    )

    bootstrap_iters: int = field(
        default=0,
        metadata={
            "help": (
                "Bootstrap iterations for lm_eval stderr estimates. Set to 0 to "
                "skip stderr bootstrapping during training."
            )
        },
    )

    cache_requests: bool = field(
        default=False,
        metadata={
            "help": (
                "Cache lm_eval task requests. This does not cache model outputs "
                "and is safe across submodel profiles."
            )
        },
    )

    log_samples: bool = field(
        default=False,
        metadata={"help": "Whether lm_eval should retain per-sample outputs in its result."},
    )

    suppress_output: bool = field(
        default=True,
        metadata={
            "help": "Suppress lm_eval stdout to reduce log spam (set to False for debugging)"
        },
    )


@dataclass
class TrainArguments(SerializableMixin):
    """Hydra-compatible subset of ``transformers.TrainingArguments``."""

    output_dir: str = field(
        default=f"./outputs/{NOW}",
        metadata={
            "help": (
                "The output directory where the model predictions and checkpoints will be written"
            )
        },
    )

    disable_tqdm: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the tqdm progress bars."},
    )

    log_level: str = field(
        default="info",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels"
                "as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a 'passive'"
                "level which doesn't set anything and lets the application set the level."
                "Defaults to 'passive'."
            ),
            "options": list(trainer_log_levels.keys()),
        },
    )

    logging_strategy: str = field(
        default="steps",
        metadata={
            "help": "The logging strategy to use",
            "options": ["no", "steps", "epoch"],
        },
    )
    per_device_train_batch_size: int = field(
        default=omegaconf.MISSING,
        metadata={"help": "Batch size per device accelerator core/CPU for training."},
    )

    per_device_eval_batch_size: int = field(
        default=omegaconf.MISSING,
        metadata={"help": "Batch size per device accelerator core/CPU for evaluation."},
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing an update pass."
            "Remind that the total batch size is given by"
            "(per_device_train_batch_size x gradient_accumulation_steps x num_devices)"
        },
    )

    bf16: bool = field(
        default=True,
        metadata={"help": ("Whether to use bf16 (mixed) precision instead of 32-bit.")},
    )

    eval_strategy: str = field(
        default="steps",
        metadata={
            "help": "The evaluation strategy to use",
            "options": ["no", "steps", "epoch"],
        },
    )

    eval_steps: Optional[float] = field(
        default=omegaconf.MISSING,
        metadata={
            "help": "Run an evaluation every X steps. Should be an integer or a float in range"
            "`[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps."
        },
    )

    learning_rate: float = field(
        default=omegaconf.MISSING,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    weight_decay: float = field(
        default=omegaconf.MISSING,
        metadata={"help": "Weight decay for AdamW if we apply some."},
    )

    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})

    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})

    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})

    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm, used for gradient clipping"}
    )

    max_steps: int = field(
        default=omegaconf.MISSING, metadata={"help": "Total number of training steps"}
    )

    lr_scheduler_type: str = field(
        default="constant", metadata={"help": "The scheduler type to use."}
    )

    lr_scheduler_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": ("Extra parameters for the lr_scheduler.")},
    )
    warmup_steps: int = field(
        default=omegaconf.MISSING, metadata={"help": "Linear warmup over warmup_steps."}
    )

    log_on_each_node: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log "
                "once per node or just once on the main node"
            )
        },
    )

    logging_steps: float = field(
        default=10,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )

    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )

    data_seed: Optional[int] = field(
        default=None, metadata={"help": "Random seed to be used with data samplers."}
    )

    run_name: str = field(
        default=NOW,
        metadata={
            "help": "This is just to avoid the wandb warning, we already set the name of the"
            "run via `WandbArguments`"
        },
    )

    report_to: Optional[str] = field(
        default=None,
        metadata={"help": "The list of integrations to report the results and logs to."},
    )

    save_strategy: str = field(
        default="steps",
        metadata={
            "help": "The checkpoint save strategy to use.",
            "options": ["no", "steps"],
        },
    )

    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range"
                "`[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )

    save_total_limit: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints."
                "Deletes the older checkpoints in `output_dir`. The HF default is "
                "unlimited checkpoints, here we overwrite it to 1"
            )
        },
    )

    remove_unused_columns: Optional[bool] = field(default=False)
    dataloader_num_workers: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only)."
                "0 means that the data will be loaded in the main process. Defaults to 0."
            )
        },
    )
    dataloader_persistent_workers: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If True, the data loader will not shut down the worker processes after a dataset"
                "has been consumed once. This allows to maintain the workers Dataset instances "
                "alive. Can potentially speed up training, but will increase RAM usage."
                "Default to False."
            )
        },
    )

    ddp_find_unused_parameters: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether DDP should traverse the autograd graph looking for unused "
                "parameters. FlexRank uses all trainable parameters in the training path, "
                "so disabling this avoids extra per-step overhead."
            )
        },
    )

    torch_compile: bool = field(
        default=False,
        metadata={"help": "Compile the model using `torch.compile()` for faster training."},
    )

    torch_compile_backend: str | None = field(
        default=None,
        metadata={
            "help": "Backend for `torch.compile()`. If set, automatically enables `torch_compile`.",
        },
    )

    torch_compile_mode: str | None = field(
        default=None,
        metadata={
            "help": "Compilation mode for `torch.compile()`. "
            "If set, automatically enables `torch_compile`.",
        },
    )


@dataclass
class DistillArguments:
    """Additional arguments needed to run distillation training."""

    ce_loss_w: float = field(
        default=omegaconf.MISSING,
        metadata={"help": "Weight of the cross-entropy function for the clm or mlm loss"},
    )

    kl_loss_w: float = field(
        default=omegaconf.MISSING,
        metadata={"help": "Weight of the KL-loss for distillation"},
    )

    temperature: float = field(default=1.0, metadata={"help": "Temperature used in the KL-loss"})

    def __post_init__(self):
        assert round(self.ce_loss_w + self.kl_loss_w, 1) == 1.0, (
            f"Losses should sum up to 1, got {self.ce_loss_w=} and {self.ce_loss_w=}"
        )


@dataclass
class DistillTrainArguments(TrainArguments, DistillArguments):
    """Complete set of arguments for running distillation training."""

    _export_fields = [
        "ce_loss_w",
        "kl_loss_w",
        "temperature",
        "adam_beta1",
        "adam_beta2",
        "max_grad_norm",
        "learning_rate",
        "weight_decay",
        "lr_scheduler_type",
        "warmup_steps",
        "bf16",
    ]


@dataclass
class DistillationTrainingArguments(TrainingArguments, DistillArguments):
    """TrainingArguments variant that adds the distillation fields."""


class TaskName(StrEnum):
    """Task names supported by the system."""

    VISION = auto()
    NLP = auto()


@dataclass
class Config(SerializableMixin):
    """Top-level FlexRank training and evaluation configuration."""

    task: TaskName
    train: DistillTrainArguments
    eval: TrainArguments
    lm_eval: LMEvalArguments
    model: BaseModelArguments
    dataset: BaseDataArguments
    decomposition: DecompositionArguments
    profile_algo: ProfileSearchAlgoArguments
    sampler: SamplerArguments
    logger: WandbArguments
    last_profile_algo: ProfileSearchAlgoArguments

    load_flexrank_path: Optional[str] = None
    resume_checkpoint_path: Optional[str | bool] = False

    @property
    def hf_train(self) -> DistillationTrainingArguments:
        """Convert DistillTrainArguments into HF-compatible arguments"""
        return DistillationTrainingArguments(**vars(self.train), label_names=["labels"])

    @property
    def hf_eval(self) -> TrainingArguments:
        """Convert TrainArguments into HF-compatible arguments"""
        return TrainingArguments(**vars(self.eval), label_names=["labels"])

    def _check_load_path(self):
        assert (self.load_flexrank_path is None) or Path(self.load_flexrank_path).exists(), (
            f"Path {self.load_flexrank_path} does not exist"
        )

    def _check_task_consistency(self):
        # Validate consistency task <-> (model, dataset)
        if self.task is TaskName.NLP:
            req_types = (NLPModelArguments, NLPDataArguments)
        else:  # guaranteed to be 'vision'
            req_types = (VisionModelArguments, VisionDataArguments)

        to_check = (self.model, self.dataset)
        msg = (
            f"Task '{self.task}' requires type(model)=NLPModelArguments "
            "and type(dataset)=NLPDataArguments, got: "
            f"type(model)={to_check[0].__class__.__name__} and "
            f"type(dataset)={to_check[1].__class__.__name__}"
        )

        assert all(isinstance(var, req_t) for var, req_t in zip(to_check, req_types)), msg

    def _check_eval_args(self):
        assert not self.eval.torch_compile, (
            "Evaluation/calibration trainers must run without torch_compile. "
            "FlexRank DataSVD collects activations with forward hooks, which are "
            "not reliable through compiled model wrappers."
        )
        assert self.eval.torch_compile_backend is None, (
            "Evaluation/calibration trainers must not set torch_compile_backend."
        )
        assert self.eval.torch_compile_mode is None, (
            "Evaluation/calibration trainers must not set torch_compile_mode."
        )

    def __post_init__(self):
        self._check_load_path()
        self._check_task_consistency()
        self._check_eval_args()


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="train", name="base_train", node=DistillTrainArguments)
cs.store(group="eval", name="base_eval", node=TrainArguments)

# For vision task
cs.store(group="model", name="nlp_model", node=NLPModelArguments)
cs.store(group="dataset", name="nlp_dataset", node=NLPDataArguments)

# For nlp task
cs.store(group="model", name="vision_model", node=VisionModelArguments)
cs.store(group="dataset", name="vision_dataset", node=VisionDataArguments)

cs.store(group="decomposition", name="base_decomposition", node=DecompositionArguments)
cs.store(group="profile_algo", name="base_search_algo", node=ProfileSearchAlgoArguments)
cs.store(group="sampler", name="base_sampler", node=SamplerArguments)
cs.store(group="lm_eval", name="base_lm_eval", node=LMEvalArguments)
cs.store(group="logger", name="base_logger", node=WandbArguments)
cs.store(group="last_profile_algo", name="base_search_algo", node=ProfileSearchAlgoArguments)


def parse_structured_config(
    version_base: str | None,
    config_path: str,
    config_name: str,
) -> Callable[[Callable[[_ConfigT], _ReturnT]], Callable[[], _ReturnT]]:
    """Parse Hydra's DictConfig into a structured config object."""
    # This is done because configs files live in the parent directory
    fixed_cfg_path = os.path.join("..", config_path)

    def decorator(func: Callable[[_ConfigT], _ReturnT]) -> Callable[[], _ReturnT]:
        @hydra.main(
            version_base=version_base,
            config_path=fixed_cfg_path,
            config_name=config_name,
        )
        def _reassign_configs(cfg: omegaconf.DictConfig) -> _ReturnT:
            structured_cfg = cast(_ConfigT, OmegaConf.to_object(cfg))
            return func(structured_cfg)

        return cast(Callable[[], _ReturnT], _reassign_configs)

    return decorator


def _main(cfg: Config) -> None:
    """Dummy entrypoint to test the config resolution"""
    assert isinstance(cfg, Config), "Failed conversion from OmegaConf.DictConfig to Config"
    print(OmegaConf.to_yaml(cfg))


_entrypoint: Any = parse_structured_config(
    version_base=None, config_path="../config", config_name="config"
)(_main)


def main() -> None:
    """Run the Hydra entrypoint."""
    _entrypoint()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
