"""Trainer implementations used by `flexrank`.

Most trainers in this package are lightweight, single-worker trainers focused on
computer-vision (CV) workflows and are primarily intended for notebooks and
interactive experimentation.

`SVDDecompositionTrainer` is the notable exception: it is task-agnostic and can
run decomposition with HuggingFace-like evaluators, including distributed
workflows through `accelerate`.
"""

from .base_trainer import (
    AbstractEvaluator,
    AbstractTrainer,
    TrainOutput,
    PredictionOutput,
    EvalLoopOutput,
    BaseEvaluator,
    BaseTrainer,
)
from .standard_trainer import TrainingArguments, StandardTrainer
from .flexrank_trainer import FlexRankTrainer, FlexRankTrainArguments
from .svd_trainer import DecompositionOutput, SVDType, SVDTrainer, SVDTrainingArguments

__all__ = [
    "AbstractEvaluator",
    "AbstractTrainer",
    "TrainOutput",
    "PredictionOutput",
    "EvalLoopOutput",
    "TrainingArguments",
    "BaseEvaluator",
    "BaseTrainer",
    "StandardTrainer",
    "FlexRankTrainer",
    "FlexRankTrainArguments",
    "DecompositionOutput",
    "SVDType",
    "SVDTrainer",
    "SVDTrainingArguments",
]
