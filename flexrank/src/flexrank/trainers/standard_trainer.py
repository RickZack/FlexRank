"""Standard single-worker CV trainer.

`StandardTrainer` provides a small HF-like training/evaluation interface for
image-classification style workflows. It is designed for single-process usage
and is primarily consumed by project notebooks.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TypeAlias

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from flexrank.utils.logger import init_logger

from .base_trainer import BaseTrainer, EvalArguments, EvalLoopOutput, TrainOutput

logger = init_logger(__name__)

__all__ = ["TrainingArguments", "StandardTrainer"]


@dataclass
class TrainingArguments(EvalArguments):
    """Minimal training configuration used by `StandardTrainer`."""

    num_train_epochs: int = 1
    logging_steps: int = 10_000
    eval_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    max_grad_norm: Optional[float] = None
    per_device_train_batch_size: int = 64
    warmup_steps: int = 0
    start_factor: float = 0.1


LRScheduler: TypeAlias = torch.optim.lr_scheduler.LRScheduler
Optimizer: TypeAlias = torch.optim.lr_scheduler.LRScheduler


@dataclass(init=False)
class TrainLoopStats:
    """Running statistics for a single epoch"""

    loss: float = 0.0
    correct: int = 0
    test_loss: int
    n_batches: int = 0
    n_samples: int = 0
    test_correct: int = 0

    @property
    def train_loss(self):
        """Running training loss"""
        return self.loss / self.n_batches

    @property
    def train_acc(self):
        """Running training accuracy"""
        return 100.0 * self.correct / self.n_samples

    def update(self, train_loss: float, train_correct: int, n_samples: int):
        """Update the running statistics"""
        self.loss += train_loss
        self.correct += train_correct
        self.n_batches += 1
        self.n_samples += n_samples


@dataclass
class StandardTrainer(BaseTrainer):
    """HF-like trainer for single-worker CV tasks.

    The class assumes one process/worker and dataloader-driven CV batches.
    """

    args: TrainingArguments = field(default_factory=TrainingArguments)
    optimizer_cls: type[Optimizer] = field(default=torch.optim.SGD)
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    scheduler_cls: type[LRScheduler] = field(default=torch.optim.lr_scheduler.ConstantLR)
    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    optimizer: Optional[torch.optim.Optimizer] = field(init=False, default=None)
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Post-init processing to set up the optimizer and scheduler."""
        epoch_steps = len(self.train_dataset) // self.args.per_device_train_batch_size
        total_steps = self.args.num_train_epochs * epoch_steps
        self.create_optimizer_and_scheduler(num_training_steps=total_steps)

    # Pylint's default argument-count threshold is too small for this trainer API.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """Create the optimizer and optional warmup/scheduler chain."""
        _ = num_training_steps
        optimizer_cls = self.optimizer_cls
        optimizer_kwargs = self.optimizer_kwargs
        scheduler_cls = self.scheduler_cls
        scheduler_kwargs = self.scheduler_kwargs

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
        schedulers = [scheduler_cls(self.optimizer, **scheduler_kwargs)]
        milestones: list[int] = []
        if self.args.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.args.start_factor,
                total_iters=self.args.warmup_steps,
            )
            schedulers.insert(0, warmup_scheduler)
            milestones.append(self.args.warmup_steps)

        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers, milestones
        )

    def _train_dataloader_iterator(self):
        train_loader = self._get_train_dataloader()
        device = self.args.distr.device
        return ((d.to(device), t.to(device)) for d, t in train_loader)

    def _get_train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "A train dataset must be provided."
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            shuffle=True,
        )

    def _forward_backward(
        self, data: Tensor, target: Tensor, criterion: torch.nn.Module
    ) -> tuple[Tensor, Tensor]:
        output = self.model(data)
        loss = criterion(output, target)
        loss.backward()

        return output, loss.detach()

    def _clip_grad_norm(self):
        if self.args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

    def _get_target_labels(self, target: Tensor) -> Tensor:
        """Return class labels used to compute batch accuracy."""
        return target

    def _on_train_epoch_end(self) -> None:
        """Hook for subclasses that need epoch-end cleanup."""

    def train_epoch(self, epoch: int, criterion: torch.nn.Module) -> float:
        """Run a single training epoch and return the mean loss."""
        assert self.optimizer is not None, "Optimizer must be created before training."
        assert self.lr_scheduler is not None, "Scheduler must be created before training."
        self.model.train()
        stat = TrainLoopStats()
        for batch_idx, (data, target) in enumerate(self._train_dataloader_iterator(), start=1):
            self.optimizer.zero_grad()
            output, loss = self._forward_backward(data, target, criterion)

            self._clip_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()

            pred = output.data.max(1, keepdim=True)[1]
            target_labels = self._get_target_labels(target).view_as(pred)
            correct = pred.eq(target_labels).sum().item()
            stat.update(loss.item(), correct, len(data))

            if not batch_idx % self.args.logging_steps:
                logger.info(
                    "Train Epoch: %s [%s/%s (%.0f%%)]\tLoss: %.6f\tAccuracy: %.2f%%",
                    epoch,
                    batch_idx * len(data),
                    len(self.train_dataset),
                    100.0 * batch_idx / len(self.train_dataset),
                    stat.train_loss,
                    stat.train_acc,
                )

        self._on_train_epoch_end()
        return stat.train_loss, stat.train_acc

    @torch.no_grad()
    # Pylint's default argument-count threshold is too small for this HF-like API.
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def evaluation_loop(
        self,
        dataloader: Optional[DataLoader] = None,
        description: Optional[str] = None,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Evaluate the current model on a dataloader and return aggregate metrics."""
        _ = description, prediction_loss_only, ignore_keys
        eval_dataloader = dataloader if dataloader is not None else self.get_eval_dataloader()
        criterion = torch.nn.CrossEntropyLoss()
        device = self.args.distr.device
        self.model.eval()
        total_loss = 0.0
        correct = 0
        num_samples = len(eval_dataloader.dataset)
        for data, target in eval_dataloader:
            data, target = data.to(device), target.to(device)
            output = self.model(data)
            total_loss += criterion(output, target).item() * data.shape[0]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        eval_loss = total_loss / num_samples
        eval_acc = 100.0 * correct / num_samples
        metrics = {
            f"{metric_key_prefix}_accuracy": eval_acc,
            f"{metric_key_prefix}_loss": eval_loss,
        }
        return EvalLoopOutput(
            predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples
        )

    def _do_validation(self, epoch: int) -> None:
        eval_strategy = self.args.eval_strategy
        return (
            eval_strategy == "epoch"
            or (epoch - 1) * len(self.train_dataset) >= self.args.eval_steps
        )

    # pylint: disable=too-many-locals
    def train(
        self,
        resume_from_checkpoint: Optional[str | bool] = None,
        trial: Any = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
    ) -> TrainOutput:
        """Train for the configured number of epochs and return summary metrics."""
        unsupported_args = (
            resume_from_checkpoint,
            trial,
            ignore_keys_for_eval,
        )
        assert all(arg is None for arg in unsupported_args), (
            "resume_from_checkpoint, trial, and ignore_keys_for_eval are not supported."
        )

        self.model.to(self.args.distr.device)
        accuracy: list[float] = []
        losses: list[float] = []
        epoch_losses: list[float] = []
        epoch_accuracies: list[float] = []
        criterion = torch.nn.CrossEntropyLoss()
        n_epochs = self.args.num_train_epochs
        for epoch in trange(1, n_epochs + 1, desc=self.__class__.__name__, leave=False):
            with logging_redirect_tqdm():
                train_loss, train_acc = self.train_epoch(epoch, criterion)
                epoch_losses.append(train_loss)
                epoch_accuracies.append(train_acc)
                if self._do_validation(epoch):
                    eval_metrics = self.evaluate(
                        eval_dataset=self.eval_dataset, metric_key_prefix="eval"
                    )
                    accuracy.append(float(eval_metrics["eval_accuracy"]))
                    losses.append(float(eval_metrics["eval_loss"]))
        global_step = n_epochs * len(self.train_dataset)
        training_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else 0.0
        metrics = {
            "train_loss": training_loss,
            "train_loss_history": epoch_losses,
            "train_accuracy_history": epoch_accuracies,
        }
        if losses:
            metrics["eval_loss"] = float(losses[-1])
            metrics["eval_accuracy"] = float(accuracy[-1])
            metrics["eval_loss_history"] = list(losses)
            metrics["eval_accuracy_history"] = list(accuracy)
        return TrainOutput(global_step=global_step, training_loss=training_loss, metrics=metrics)
