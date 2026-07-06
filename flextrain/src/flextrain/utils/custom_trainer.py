"""Custom Trainer properties shared by FlexTrain components."""

from transformers import Trainer

from flexrank.types import DistributedInfo

from .utils import get_distributed_info


class CustomTrainer(Trainer):
    """HF-Transformer trainer that has the `distr` property."""

    @property
    def distr(self) -> DistributedInfo:
        """Return distributed runtime information for the trainer accelerator."""
        return get_distributed_info(self.accelerator)

    @property
    def num_model_parameters(self) -> int:
        """Return the total number of model parameters."""
        return sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def num_train_model_parameters(self) -> int:
        """Return the number of trainable model parameters."""
        return sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
