"""Callback for computing profiles after training finishes."""

import os
import weakref
from typing import TypeAlias, Union

from transformers import Trainer, TrainerCallback

from flexrank.utils import init_logger
from flextrain.dataset import DatasetSplits
from flextrain.utils import get_distributed_info
from flextrain.utils.args import ProfileSearchAlgoArguments
from flextrain.utils.flexrank_data import FlexRankData
from flextrain.utils.plot import FlexRankProfilePlotConfig, save_flexrank_profile_plot

__all__ = ["PostTrainProfileCallback"]

TrainerProxy: TypeAlias = Union[Trainer, weakref.ProxyType]
FlexRankDataProxy: TypeAlias = Union[FlexRankData, weakref.ProxyType]

logger = init_logger(__name__)


class PostTrainProfileCallback(TrainerCallback):
    """Run a final profile search and save its outputs at train end."""

    def __init__(
        self,
        trainer: Trainer,
        flexdata: FlexRankData,
        splits: DatasetSplits,
        n_models: int,
        min_p: float,
        profile_algo: ProfileSearchAlgoArguments,
    ):
        self.trainer: TrainerProxy = weakref.proxy(trainer)
        self.flexdata: FlexRankDataProxy = weakref.proxy(flexdata)
        self.ds_splits = splits
        self.n_models = n_models
        self.min_p = min_p
        self.profile_algo = profile_algo

    def on_train_end(self, args, state, control, **kwargs):
        """Compute, evaluate, plot, and persist final profile data."""
        from flextrain.utils.flexrank_utils import get_submodel_profiles  # pylint: disable=import-outside-toplevel

        # Final profiling logic
        if self.profile_algo.classname:
            self.flexdata.trainer.eval_dataset = self.ds_splits.calib
            pred_stats, sol = get_submodel_profiles(
                self.profile_algo,
                self.trainer,
                self.n_models,
                self.min_p,
            )

            self.flexdata.trainer.eval_dataset = self.ds_splits.val
            distr = get_distributed_info(self.trainer.accelerator)
            submodels_stats = pred_stats.evaluate_profiles(self.trainer, distr=distr)

            self.flexdata.sol = sol
            self.flexdata.last_profiles_data = pred_stats
            self.flexdata.end_profiles_loss = submodels_stats.metrics

            if self.trainer.accelerator.is_main_process:
                self.flexdata.save()
                plots_dir = os.path.join(self.trainer.args.output_dir, "plots")
                save_flexrank_profile_plot(
                    plots_dir,
                    f"step-{state.step}.png",
                    [
                        (
                            submodels_stats.params,
                            submodels_stats.metrics,
                            f"Step {state.step}",
                        )
                    ],
                    FlexRankProfilePlotConfig(
                        full_size=self.flexdata.full_model_parameters,
                        full_loss=self.flexdata.full_model_loss,
                    ),
                )

        return control
