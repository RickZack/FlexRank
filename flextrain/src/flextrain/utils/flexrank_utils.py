"""Helpers for building and restoring FlexRank training workflows."""

from copy import deepcopy

import numpy as np
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from flexrank.profiles import (
    MergedProfilesData,
    ProfileAlgoSolution,
    ProfilesData,
    ProfileSearchAlgo,
    get_od_layers,
    get_profile_search_algo,
    inner_dims_profile_to_params,
)
from flexrank.trainers import AbstractEvaluator, SVDTrainer, SVDTrainingArguments
from flexrank.utils import init_logger, set_distributed_logging_state
from flextrain.callbacks import add_flexrank_callbacks
from flextrain.dataset import CollatorType, DatasetSplitsWithCollator, DatasetType
from flextrain.model import (
    FlexRankModel,
    ModelWithProcessorAndMetric,
    replace_conv1d_with_linear,
)
from flextrain.utils import get_distributed_info
from flextrain.utils.flexrank_data import FlexRankData

from .args import (
    Config,
    DecompositionArguments,
    DistillationTrainingArguments,
    ProfileSearchAlgoArguments,
)
from .custom_trainer import CustomTrainer
from .distillation_trainer import DistillationTrainer
from .utils import suppress_stdout

log = init_logger(__name__)


def init_eval_trainer(
    model_data: ModelWithProcessorAndMetric,
    collator: CollatorType,
    eval_dataset: DatasetType,
    eval_args: TrainingArguments,
    *,
    deepcopy_model: bool = True,
) -> CustomTrainer:
    """Build an evaluation-only trainer used for calibration and profiling."""
    assert eval_args.max_steps <= 1, "Eval trainer should at most max_steps=1"
    assert not eval_args.learning_rate, "Eval trainer should have learning_rate=0"

    model = deepcopy(model_data.model) if deepcopy_model else model_data.model
    trainer = CustomTrainer(
        model=model,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=model_data.proc,
        args=eval_args,
        compute_metrics=model_data.metric_fn,
    )
    # Dummy call to train to avoid bug with FSDP on evaluate()
    trainer.train()
    return trainer


@suppress_stdout
def init_flexrank_model(
    model_data: ModelWithProcessorAndMetric,
    calib_trainer: AbstractEvaluator,
    args: Config,
    *,
    deepcopy_model: bool = True,
) -> tuple[ModelWithProcessorAndMetric, int]:
    """
    Initialize FlexRank model via SVD decomposition and wrap in FlexRankModel.

    Takes a model, applies SVD-based low-rank decomposition using calibration data,
    and wraps the result in a FlexRankModel for saving/loading.

    Args:
        model_data: ModelWithProcessorAndMetric containing base model
        calib_trainer: Trainer with calibration data for SVD
        args: DecompositionArguments with decomposition settings
        deepcopy_model: Whether to deepcopy model before decomposition

    Returns:
        Tuple of (ModelWithProcessorAndMetric with FlexRankModel wrapper, num_params)
    """
    model = deepcopy(model_data.model) if deepcopy_model else model_data.model
    svd_args = SVDTrainingArguments(
        args.decomposition.svd_type,
        args.decomposition.exclude_layers_names,
        gram_cache_device=args.decomposition.gram_cache_device,
        distr=calib_trainer.distr,
    )

    svd_trainer = SVDTrainer(model, calib_trainer, svd_args)
    all_decomp_layers = svd_trainer.train().decomposed_layers

    # Wrap the SVD-decomposed model in FlexRankModel
    model = FlexRankModel.from_model(
        svd_trainer.model, decomposed_layer_names=all_decomp_layers, training_args=args
    )

    flex_model = ModelWithProcessorAndMetric(model, model_data.proc, model_data.metric_fn)

    return flex_model, svd_trainer.num_model_parameters


def init_flexrank_trainer(
    flex: ModelWithProcessorAndMetric,
    args: DistillationTrainingArguments,
    base: ModelWithProcessorAndMetric,
    splits: DatasetSplitsWithCollator,
) -> DistillationTrainer:
    """Build the distillation trainer for the decomposed FlexRank model."""
    assert flex.model is not base.model, (
        "The model to be trained must be different than the teacher model"
    )
    teacher_model = None
    if args.kl_loss_w > 0:
        teacher_model = base.model

    trainer = DistillationTrainer(
        model=flex.model,
        args=args,
        train_dataset=splits.train,
        eval_dataset=splits.val,
        data_collator=splits.collator,
        processing_class=flex.proc,
        teacher_model=teacher_model,
        compute_metrics=flex.metric_fn,
    )

    return trainer


@suppress_stdout
def get_submodel_profiles(
    args: ProfileSearchAlgoArguments,
    calib_trainer: AbstractEvaluator,
    n_models: int,
    min_p: float,
) -> tuple[ProfilesData, ProfileAlgoSolution]:
    """Search submodel profiles and return both the profiles and solver result."""

    def get_param_thresholds():
        """Compute evenly spaced parameter-saving thresholds."""

        od_layers = get_od_layers(calib_trainer.model)
        min_profile = [int(np.ceil(min_p * layer.max_inner_dim)) for layer in od_layers]
        full_profile = [layer.max_inner_dim for layer in od_layers]
        min_params = inner_dims_profile_to_params(min_profile, od_layers)
        max_saving = inner_dims_profile_to_params(full_profile, od_layers) - min_params
        return np.linspace(0, max_saving, n_models)

    algo: ProfileSearchAlgo = get_profile_search_algo(
        args.classname, calib_trainer, algo_kwargs=args.algo_kwargs
    )
    sol = algo.solve()
    param_thresholds = get_param_thresholds()
    profiles_data = sol.thresholded(param_thresholds, eager_pruning=args.eager_pruning)

    return profiles_data, sol


def _get_flexrank_from_path(
    args: Config,
    model_data: ModelWithProcessorAndMetric,
    splits: DatasetSplitsWithCollator,
) -> FlexRankData:
    """Restore a previously saved FlexRank run and attach callbacks."""
    checkpoint_path = get_last_checkpoint(args.load_flexrank_path)
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found in {args.load_flexrank_path}")

    flex_model = ModelWithProcessorAndMetric(
        FlexRankModel.from_pretrained(checkpoint_path),
        model_data.proc,
        model_data.metric_fn,
    )

    trainer = init_flexrank_trainer(flex_model, args.hf_train, model_data, splits)
    flex_data = FlexRankData.load(input_dir=args.load_flexrank_path)
    flex_data.trainer = trainer
    add_flexrank_callbacks(trainer, flex_data, splits, args)

    return flex_data


def _get_eval_profiles(
    prof_algo: ProfileSearchAlgoArguments,
    decomp: DecompositionArguments,
    calib_tr: AbstractEvaluator,
    eval_ds: DatasetType,
) -> tuple[ProfileAlgoSolution, MergedProfilesData]:
    """Evaluate discovered profiles on calibration and validation data."""
    pred_stats, sol = get_submodel_profiles(prof_algo, calib_tr, decomp.n_models, decomp.min_p)
    distr = get_distributed_info(calib_tr.accelerator)
    calib_stats = pred_stats.evaluate_profiles(
        calib_tr, distr=distr, metric_key_prefix="calib", verbose=True
    )
    eval_stats = pred_stats.evaluate_profiles(calib_tr, eval_ds, distr=distr, verbose=True)
    return sol, MergedProfilesData.from_profiles_data(pred_stats, calib_stats, eval_stats)


@suppress_stdout
def get_flexrank(
    args: Config,
    model_data: ModelWithProcessorAndMetric,
    splits: DatasetSplitsWithCollator,
) -> FlexRankData:
    """Create or restore FlexRank state, then attach training callbacks."""
    replace_conv1d_with_linear(model_data.model)
    if args.load_flexrank_path is not None:
        return _get_flexrank_from_path(args, model_data, splits)

    eval_args = args.hf_eval
    decomp_args = args.decomposition
    calib_tr = init_eval_trainer(model_data, splits.collator, splits.calib, eval_args)
    set_distributed_logging_state(calib_tr.distr)
    model_loss = calib_tr.evaluate()["eval_loss"]

    # model has not been modified by FSDP and it is on cpu
    flex_model, n_params = init_flexrank_model(model_data, calib_tr, args)

    calib_tr = init_eval_trainer(flex_model, splits.collator, splits.calib, eval_args)
    flex_loss = calib_tr.evaluate()["eval_loss"]
    log.info("Loss before SVD: %.4f - after SVD: %.4f", model_loss, flex_loss)
    sol, prof_data = _get_eval_profiles(args.profile_algo, decomp_args, calib_tr, splits.val)
    flex_model.model.profiles_data = sol.to_profiles_data()
    train_args = args.hf_train
    trainer = init_flexrank_trainer(flex_model, train_args, model_data, splits)
    flex_data = FlexRankData(trainer, prof_data, sol, n_params, model_loss, decomp_args)
    flex_data.save()

    add_flexrank_callbacks(trainer, flex_data, splits, args)
    return flex_data
