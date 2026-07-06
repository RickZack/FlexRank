"""Samplers backed by an explicit list of predefined profiles."""

import numpy as np
import torch

from flexrank.trainers.base_trainer import BaseEvaluator

from .base_sampler import BaseSampler
from .registry import register_sampler

__all__ = ["PredefinedModelsSampler", "PredefinedModelsCurriculumSampler"]


@register_sampler
class PredefinedModelsSampler(BaseSampler):
    """Sample from a fixed list of predefined model profiles."""

    def __init__(self, *args, sandwich: bool = False, **kwargs):
        """Initialize the predefined-profile sampler."""
        super().__init__(*args, **kwargs)
        self.smallest_model = self._samples[-1]
        self.largest_model = self._samples[0]
        self.sandwich = sandwich

    def sampler(self):
        """Yield predefined profiles in random order."""
        while True:
            random_perm = np.random.permutation(len(self._samples))
            for idx in random_perm:
                sample = self._samples[idx]
                if self.sandwich:
                    yield self.largest_model
                    yield sample
                    yield self.smallest_model
                else:
                    yield sample


@register_sampler
class PredefinedModelsCurriculumSampler(BaseSampler):
    """Adapt sampling probabilities based on recent submodel loss changes."""

    def __init__(
        self,
        evaluator: BaseEvaluator,
        *args,
        period: int = 1,
        warmup_steps: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize the curriculum sampler."""
        super().__init__(*args, **kwargs)
        self.period = period
        self.warmup_steps = warmup_steps or len(self._samples)
        self.evaluator = evaluator
        self.cur_it = 0
        self.old_loss = self._evaluate_submodels_loss()
        self.frequencies = np.zeros(len(self.old_loss), dtype=np.int32)

    def _eval_model(self) -> float:
        """Evaluate the currently active profile and return its loss."""
        return self.evaluator.evaluate()["eval_loss"]

    @torch.no_grad()
    def _evaluate_submodels_loss(self) -> np.ndarray:
        """Evaluate every predefined profile and collect their losses."""
        num_submodels = len(self._samples)
        losses = np.empty(num_submodels)
        self._model.eval()
        for idx in range(num_submodels):
            self.set_p_for_layers(self._samples[idx])
            loss = self._eval_model()
            losses[idx] = loss
        self._model.train()  # put back the model on train mode
        return losses

    def _calculate_fresh_sampling(self) -> int:
        """Recompute losses and pick the profile with the smallest improvement."""
        new_loss = self._evaluate_submodels_loss()
        delta = self.old_loss - new_loss
        self.old_loss = new_loss
        return np.argmin(delta)

    def _predict_next_sampling(self) -> int:
        """Sample the next profile index from the learned frequency distribution."""
        probs = self.sampling_probabilities
        index = np.random.choice(len(self._samples), p=probs)
        return index

    def sampler(self):
        """Yield profiles using the curriculum update schedule."""
        while True:
            update_sampling = not self.cur_it % self.period
            is_warmup_iter = self.cur_it < self.warmup_steps

            if is_warmup_iter or update_sampling:
                index = self._calculate_fresh_sampling()
                self.frequencies[index] += 1
            else:
                index = self._predict_next_sampling()

            self.cur_it += 1
            yield self._samples[index]

    @property
    def sampling_probabilities(self) -> np.ndarray:
        """Return the empirical sampling distribution accumulated so far."""
        return self.frequencies / self.frequencies.sum()
