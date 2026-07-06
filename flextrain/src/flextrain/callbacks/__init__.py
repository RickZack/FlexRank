"""Callback wiring helpers for FlexRank training workflows."""

from transformers import Trainer

from flexrank.samplers import get_sampler_class_by_name
from flextrain.callbacks.eval_callback import SubmodelEvalCallback
from flextrain.callbacks.lm_eval_callback import LMEvalCallback
from flextrain.callbacks.post_train_profile_callback import PostTrainProfileCallback
from flextrain.callbacks.sampler_callback import SamplerCallback
from flextrain.dataset import DatasetSplits
from flextrain.utils.args import Config, SamplerArguments
from flextrain.utils.flexrank_data import FlexRankData

__all__ = ["add_flexrank_callbacks"]


def _get_sampler_callback(
    trainer: Trainer, profiles: list[list], args: SamplerArguments
) -> SamplerCallback:
    """Build and initialize the sampler callback for the given profile set."""
    sampler_callback = SamplerCallback(trainer, profiles)
    sampler_callback.init_sampler(
        sampler_cls=get_sampler_class_by_name(args.classname),
        sampler_kwargs={"samples": profiles, **args.sampler_kwargs},
    )
    return sampler_callback


def add_flexrank_callbacks(
    trainer: Trainer, flexdata: FlexRankData, splits: DatasetSplits, args: Config
):
    """Attach the standard FlexRank callbacks to a trainer."""
    sampler_callback = _get_sampler_callback(trainer, flexdata.profiles_data.profiles, args.sampler)
    eval_callback = SubmodelEvalCallback(trainer, flexdata, splits)
    trainer.add_callback(sampler_callback)
    trainer.add_callback(eval_callback)

    if args.lm_eval.enabled:
        trainer.add_callback(LMEvalCallback(trainer, flexdata, args.lm_eval))

    if args.last_profile_algo.classname:
        post_train_profile_callback = PostTrainProfileCallback(
            trainer,
            flexdata,
            splits,
            args.decomposition.n_models,
            args.decomposition.min_p,
            args.last_profile_algo,
        )
        trainer.add_callback(post_train_profile_callback)
