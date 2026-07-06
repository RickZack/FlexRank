"""Base trainer/evaluator protocols and shared output types.

This module defines the common interfaces and lightweight base classes used by
`flexrank.trainers`.

Note:
- Concrete CV training behavior is implemented in `standard_trainer.py` and
  derived trainers, which are designed for single-worker usage and are mainly
  used by notebooks.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional, Protocol, TypeAlias

from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from flexrank.types import DistributedInfo
from flexrank.types.distributed import get_single_worker_distributed_info

from ..utils.logger import init_logger
from .utils import save_model

InputDataClass: TypeAlias = Any
DataCollator: TypeAlias = Callable[[list[InputDataClass]], dict[str, Any]]
DatasetLike: TypeAlias = Dataset | DataLoader

logger = init_logger(__name__)

__all__ = [
    "AbstractEvaluator",
    "AbstractTrainer",
    "EvalArguments",
    "TrainOutput",
    "PredictionOutput",
    "EvalLoopOutput",
    "BaseEvaluator",
    "BaseTrainer",
]


class AbstractEvaluator(Protocol):
    """Abstract evaluation protocol."""

    model: Module
    compute_metrics: Any

    @property
    def distr(self) -> DistributedInfo:
        """Distributed setup available during evaluation."""

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset | str] = None) -> DataLoader:
        """By default, we use the same dataloader for evaluation and testing."""

    def evaluate(
        self,
        eval_dataset: Optional[Dataset | dict[str, Dataset]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """Evaluate the model on the provided eval dataset and return metrics."""

    def evaluation_loop(
        self,
        dataloader: Optional[DataLoader] = None,
        description: Optional[str] = None,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> "EvalLoopOutput":
        """Run an evaluation loop and return predictions, labels, and metrics."""


class AbstractTrainer(Protocol):
    """Abstract training protocol."""

    @property
    def distr(self) -> DistributedInfo:
        """Distributed setup available during training."""

    def train(
        self,
        resume_from_checkpoint: Optional[str | bool] = None,
        trial: Any = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
    ):
        """Train the model, optionally resuming from a checkpoint."""

    @property
    def num_model_parameters(self) -> int:
        """Return the total number of model parameters."""

    @property
    def num_train_model_parameters(self) -> int:
        """Return the number of trainable model parameters."""


@dataclass
class EvalArguments:
    """Evaluation configuration used by `BaseEvaluator` and derived classes."""

    output_dir: str = "./outputs"
    eval_batch_size: int = 64
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = False
    distr: DistributedInfo = field(default_factory=get_single_worker_distributed_info)


@dataclass(frozen=True)
class TrainOutput:
    """Output of a training loop, containing global step, training loss, and metrics."""

    global_step: int
    training_loss: float
    metrics: dict[str, float]


class PredictionOutput(NamedTuple):
    """Output of a prediction loop, containing predictions, labels, and metrics."""

    predictions: Any
    label_ids: Any
    metrics: dict[str, float] | None


class EvalLoopOutput(NamedTuple):
    """Output of an evaluation loop, containing predictions, labels, and metrics."""

    predictions: Any
    label_ids: Any
    metrics: dict[str, float] | None
    num_samples: int | None


@dataclass
class BaseEvaluator(AbstractEvaluator):
    """Minimal HF-like evaluator base.

    Subclasses are expected to implement `evaluation_loop`.
    """

    model: Module
    args: EvalArguments = field(default_factory=EvalArguments)
    eval_dataset: Dataset | None = None
    compute_metrics: Any = field(init=False, default=None)

    @property
    def distr(self) -> DistributedInfo:
        """Distributed setup of the evaluator."""
        return self.args.distr

    def _get_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            shuffle=shuffle,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset | str] = None) -> DataLoader:
        ds = eval_dataset or self.eval_dataset
        assert ds is not None, "An eval dataset must be provided."
        assert not isinstance(ds, str), (
            "String dataset identifiers are not supported by "
            "BaseEvaluator; provide a Dataset or DataLoader instance."
        )
        return self._get_dataloader(ds, shuffle=False)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """By default, we use the same dataloader for evaluation and testing."""
        return self._get_dataloader(test_dataset, shuffle=False)

    # Pylint's default argument-count threshold is too small for this HF-like API.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def evaluation_loop(
        self,
        dataloader: Optional[DataLoader] = None,
        description: Optional[str] = None,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Run an evaluation loop and return predictions, labels, and metrics."""
        _ = (
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
        raise NotImplementedError

    def evaluate(
        self,
        eval_dataset: Optional[DatasetLike | dict[str, Dataset]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate a dataset or dataloader and return metric values."""
        assert eval_dataset is None or isinstance(eval_dataset, (Dataset, DataLoader)), (
            "eval_dataset must be a Dataset, DataLoader, or None."
        )
        dataloader = (
            eval_dataset
            if isinstance(eval_dataset, DataLoader)
            else self.get_eval_dataloader(eval_dataset)
        )
        return (
            self.evaluation_loop(
                dataloader=dataloader,
                description="Evaluation",
                prediction_loss_only=False,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            ).metrics
            or {}
        )

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "test",
    ) -> PredictionOutput:
        """Run evaluation on the test dataset and return predictions and metrics."""
        metrics = (
            self.evaluation_loop(
                dataloader=self.get_test_dataloader(test_dataset),
                description="Prediction",
                prediction_loss_only=False,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            ).metrics
            or {}
        )
        return PredictionOutput(predictions=None, label_ids=None, metrics=metrics)


@dataclass
class BaseTrainer(BaseEvaluator, AbstractTrainer):
    """Common trainer base exposing parameter-count helpers."""

    train_dataset: Dataset | None = None

    def train(
        self,
        resume_from_checkpoint: Optional[str | bool] = None,
        trial: Any = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
    ):
        raise NotImplementedError

    def save_model(self, output_dir: Optional[str] = None):
        """Save the model in Hugging Face's `save_pretrained` fashion."""
        output_dir = output_dir or self.args.output_dir
        save_model(self.model, output_dir)

    @property
    def num_model_parameters(self) -> int:
        return sum((p.numel() for p in self.model.parameters()))

    @property
    def num_train_model_parameters(self) -> int:
        return sum((p.numel() for p in self.model.parameters() if p.requires_grad))
