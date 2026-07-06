"""Submodel evaluation callback for FlexRank training."""

import os
import weakref
from typing import TypeAlias, Union

from transformers import Trainer, TrainerCallback

from flexrank.utils import init_logger
from flextrain.dataset import DatasetSplits
from flextrain.utils import get_distributed_info
from flextrain.utils.flexrank_data import FlexRankData
from flextrain.utils.plot import FlexRankProfilePlotConfig, save_flexrank_profile_plot

__all__ = ["SubmodelEvalCallback"]

TrainerProxy: TypeAlias = Union[Trainer, weakref.ProxyType]
FlexRankDataProxy: TypeAlias = Union[FlexRankData, weakref.ProxyType]

logger = init_logger(__name__)


class SubmodelEvalCallback(TrainerCallback):
    """Evaluate and log FlexRank submodels during Trainer evaluation."""

    def __init__(self, trainer: Trainer, flexdata: FlexRankData, splits: DatasetSplits):
        self.trainer: TrainerProxy = weakref.proxy(trainer)
        self.flexdata: FlexRankDataProxy = weakref.proxy(flexdata)
        self.ds_splits = splits

    def metrics_to_logdict(self, metrics: list[dict[str, float]], step: int):
        """Convert per-submodel metrics into Trainer log entries."""
        loss_dict = {
            f"submodels_eval/submodel_{idx}_loss": metric["eval_loss"]
            for idx, metric in enumerate(metrics, start=1)
        }
        acc_dict = {
            f"submodels_eval/submodel_{idx}_accuracy": metric["eval_accuracy"]
            for idx, metric in enumerate(metrics, start=1)
            if "eval_accuracy" in metric
        }
        avg_sub_loss = sum(loss_dict.values()) / len(loss_dict)
        avg_sub_acc = sum(acc_dict.values()) / len(acc_dict) if len(acc_dict) > 0 else None
        log_dict = {
            **loss_dict,
            **acc_dict,
            "submodels_eval/trainer_step": step,
            "submodels_eval/avg_loss": avg_sub_loss,
        }
        if avg_sub_acc is not None:
            log_dict["submodels_eval/avg_accuracy"] = avg_sub_acc
        self.trainer.log(log_dict)
        return log_dict

    def on_evaluate(self, args, state, control, **kwargs):
        """Evaluate all tracked profiles when the Trainer evaluates."""
        trainer = self.trainer
        step = state.global_step

        distr = get_distributed_info(trainer.accelerator)
        eval_profiles = self.flexdata.profiles_data.evaluate_profiles(trainer, distr=distr)
        self.flexdata.eval_history.setdefault(step, {})
        self.flexdata.eval_history[step]["eval_ds"] = eval_profiles.metrics
        self.metrics_to_logdict(eval_profiles.metrics, step)

        if step == state.max_steps:
            self.flexdata.end_profiles_loss = eval_profiles.metrics

        if trainer.accelerator.is_main_process:
            self.flexdata.save()
            plots_dir = os.path.join(trainer.args.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            save_flexrank_profile_plot(
                plots_dir,
                f"step-{step}",
                [(eval_profiles.params, eval_profiles.metrics, f"Step {step}")],
                FlexRankProfilePlotConfig(
                    full_size=self.flexdata.full_model_parameters,
                    full_loss=self.flexdata.full_model_loss,
                ),
            )

    def on_train_end(self, args, state, control, **kwargs):
        """Trigger one final evaluation at train end."""
        control.should_evaluate = True
        return control
