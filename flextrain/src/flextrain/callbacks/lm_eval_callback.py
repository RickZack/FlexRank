"""Trainer callback for evaluating FlexRank submodels with lm-eval."""
# pylint: disable=consider-using-with

import os
import weakref
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from importlib.util import find_spec
from typing import Optional, TypeAlias, Union

import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, Trainer, TrainerCallback

from flexrank.samplers.base_sampler import BaseSampler
from flexrank.utils import init_logger
from flextrain.utils.args import LMEvalArguments
from flextrain.utils.flexrank_data import FlexRankData

__all__ = ["LMEvalCallback"]

TrainerProxy: TypeAlias = Union[Trainer, weakref.ProxyType]
FlexRankDataProxy: TypeAlias = Union[FlexRankData, weakref.ProxyType]
ProcessorType: TypeAlias = Union[AutoTokenizer, AutoImageProcessor]

logger = init_logger(__name__)


class LMEvalCallback(TrainerCallback):
    """Evaluate each FlexRank submodel profile with lm-eval during training."""

    def __init__(self, trainer: Trainer, flexdata: FlexRankData, args: LMEvalArguments):
        self.trainer: TrainerProxy = weakref.proxy(trainer)
        self.flexdata: FlexRankDataProxy = weakref.proxy(flexdata)
        self.args = args
        self._last_run_step = -1
        self._disable_if_fsdp()
        self._disable_if_lm_eval_missing()

    def _disable_if_fsdp(self) -> None:
        """Disable lm-eval under FSDP, where eager full-model eval is unavailable."""
        from accelerate.utils import DistributedType  # pylint: disable=import-outside-toplevel

        if self.trainer.accelerator.distributed_type is not DistributedType.FSDP:
            return

        if self.args.enabled:
            logger.warning(
                "Disabling lm_eval callback: FSDP training is not supported because "
                "lm_eval would run on the compiled/sharded training model."
            )
        self.args.enabled = False

    def _disable_if_lm_eval_missing(self) -> None:
        """Disable lm-eval once when the optional dependency is unavailable."""
        if not self.args.enabled or find_spec("lm_eval") is not None:
            return

        logger.warning("Disabling lm_eval callback: package 'lm_eval' is not installed")
        self.args.enabled = False

    def _get_tokenizer(self) -> Optional[ProcessorType]:
        return self.trainer.processing_class

    def _get_lm_eval_model(self, model):
        """Return an eval view of the model without duplicating parameters."""
        if not getattr(self.trainer.args, "torch_compile", False):
            return model

        return self.trainer.accelerator.unwrap_model(
            model,
            keep_fp32_wrapper=True,
            keep_torch_compile=False,
        )

    def _run_lm_eval_evaluation(self, model, tokenizer):
        """Run the actual lm_eval evaluation."""
        # lm_eval is an optional dependency for users that do not enable this callback.
        from lm_eval import evaluator  # pylint: disable=import-outside-toplevel
        from lm_eval.models.huggingface import HFLM  # pylint: disable=import-outside-toplevel

        wrapped_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            batch_size=self.args.batch_size,
        )
        accelerator = self.trainer.accelerator
        wrapped_model.accelerator = accelerator
        # lm_eval reads these LM fields when splitting requests across ranks.
        wrapped_model._device = accelerator.device  # pylint: disable=protected-access
        wrapped_model._rank = accelerator.process_index  # pylint: disable=protected-access
        wrapped_model._world_size = accelerator.num_processes  # pylint: disable=protected-access

        eval_kwargs = {
            "model": wrapped_model,
            "tasks": self.args.task_names,
            "batch_size": self.args.batch_size,
            "num_fewshot": self.args.num_fewshot,
            "check_integrity": False,
            "limit": self.args.limit,
            "bootstrap_iters": self.args.bootstrap_iters,
            "cache_requests": self.args.cache_requests,
            "log_samples": self.args.log_samples,
            "verbosity": "ERROR" if self.args.suppress_output else None,
        }
        return evaluator.simple_evaluate(**eval_kwargs)

    @staticmethod
    def _format_lm_eval_results(lm_eval_result) -> dict[str, dict[str, float]]:
        if lm_eval_result is None:
            return {}

        results = {}
        task_results = lm_eval_result.get("results", {})
        for task_name, result in task_results.items():
            acc = result.get("acc,none")
            acc_stderr = result.get("acc_stderr,none")
            if acc is None:
                continue
            results[task_name] = {
                "acc": acc,
                "acc_stderr": acc_stderr if acc_stderr is not None else 0.0,
            }

        if results:
            avg_acc = sum(r["acc"] for r in results.values()) / len(results)
            results["average"] = {"acc": avg_acc}

        return results

    def _suppress_lm_eval_output(
        self,
        stack: ExitStack,
    ) -> None:
        devnull = stack.enter_context(
            open(
                os.devnull,
                "w",
                encoding="utf-8",
            )
        )
        stack.enter_context(redirect_stdout(devnull))
        stack.enter_context(redirect_stderr(devnull))

    def _evaluate_single_model(self, model, tokenizer) -> dict[str, dict[str, float]]:
        """Run lm_eval on a single model configuration."""
        was_training = model.training
        model.eval()
        try:
            with ExitStack() as stack:
                if self.args.suppress_output:
                    self._suppress_lm_eval_output(stack)
                stack.enter_context(torch.no_grad())
                lm_eval_result = self._run_lm_eval_evaluation(model, tokenizer)
        finally:
            model.train(was_training)

        return self._format_lm_eval_results(lm_eval_result)

    def _evaluate_lm_eval(self) -> list[dict[str, float]]:
        """Evaluate all submodel profiles with lm_eval."""
        trainer = self.trainer
        model = self._get_lm_eval_model(trainer.model)

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            logger.warning("Skipping lm_eval: no tokenizer/processor found in trainer")
            return []

        profiles_data = self.flexdata.profiles_data
        sampler = BaseSampler(model)
        all_results = []

        pbar = tqdm(
            zip(profiles_data.profiles, profiles_data.params),
            total=len(profiles_data.profiles),
            desc="LM-Eval submodels",
            disable=not self._is_main_process(),
        )

        for i, (profile, params) in enumerate(pbar, start=1):
            sampler.set_p_for_layers(profile)
            pbar.set_postfix(
                {
                    "submodel": f"{i}/{len(profiles_data.profiles)}",
                    "params": params,
                }
            )

            results = self._evaluate_single_model(model, tokenizer)
            all_results.append(results)

        sampler.reset_to_full()
        return all_results

    def _is_main_process(self) -> bool:
        return self.trainer.accelerator.is_main_process

    def _run_lm_eval(self, state) -> None:
        all_results = self._evaluate_lm_eval()
        if not all_results:
            return

        step = state.global_step

        if self._is_main_process():
            # Log per-submodel results
            for idx, results in enumerate(all_results, start=1):
                log_dict = {}
                for task_name, metrics in results.items():
                    if task_name == "average":
                        log_dict[f"lm_eval/submodel_{idx}_average_acc"] = metrics["acc"]
                    else:
                        log_dict[f"lm_eval/submodel_{idx}_{task_name}_acc"] = metrics["acc"]
                log_dict["lm_eval/trainer_step"] = step
                self.trainer.log(log_dict)

            # Compute and log average across all submodels
            avg_results = {}
            for task_name in all_results[0].keys():
                avg_acc = sum(r[task_name]["acc"] for r in all_results) / len(all_results)
                avg_results[task_name] = {"acc": avg_acc}

            avg_log_dict = {
                f"lm_eval/avg_{task}_acc": metrics["acc"] for task, metrics in avg_results.items()
            }
            avg_log_dict["lm_eval/trainer_step"] = step
            self.trainer.log(avg_log_dict)

            # Store in eval_history
            self.flexdata.eval_history.setdefault(step, {})
            self.flexdata.eval_history[step]["lm_eval"] = all_results
            self.flexdata.save()

        self._last_run_step = step

    def _should_run_on_step(self, step: int) -> bool:
        return (
            step > 0
            and self._last_run_step != step
            and self.args.eval_steps is not None
            and not step % self.args.eval_steps
        )

    def _should_run_on_evaluate(self) -> bool:
        return self.args.eval_steps is None

    def on_step_end(self, args, state, control, **kwargs):
        should_run = (
            self.args.enabled
            and self.args.task_names
            and self._should_run_on_step(state.global_step)
        )
        if should_run:
            self._run_lm_eval(state)

    def on_evaluate(self, args, state, control, **kwargs):
        should_run = self.args.enabled and self.args.task_names and self._should_run_on_evaluate()
        if not should_run:
            if self.args.enabled and not self.args.task_names:
                logger.warning("Skipping lm_eval: no task_names configured")
            return

        self._run_lm_eval(state)

    def on_train_end(self, args, state, control, **kwargs):
        control.should_evaluate = True
        return control
