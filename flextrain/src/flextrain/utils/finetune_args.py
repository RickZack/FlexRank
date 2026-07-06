"""Structured Hydra configuration for finetuning entry points."""

import datetime
from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers import TrainingArguments
from transformers.utils import logging

from flextrain.utils.args import (
    BaseDataArguments,
    BaseModelArguments,
    NLPDataArguments,
    NLPModelArguments,
    TaskName,
    TrainArguments,
    VisionDataArguments,
    VisionModelArguments,
    WandbArguments,
    parse_structured_config,
)

log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = {"passive": -1, **log_levels}


__all__ = ["parse_structured_config", "FinetuneConfig"]

RUN_ID = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')}"


@dataclass
class FinetuneConfig:
    """
    Configuration class for fine-tuning models.
    Attributes:
        task (TaskName): The task type (NLP or Vision) for fine-tuning.
        train (TrainArguments): Training arguments and hyperparameters.
        model (BaseModelArguments): Model-specific arguments and configuration.
        dataset (BaseDataArguments): Dataset-specific arguments and configuration.
        logger (WandbArguments): Weights & Biases logging configuration.
        resume_checkpoint_path (Optional[str | bool]): Path to checkpoint for resuming training.
            Defaults to False if not provided.
        layers_to_train (list[str]): List of layer names to train. Other layers remain frozen.
            Defaults to empty list if not provided, meaning all layers will be trained.
    Properties:
        hf_train (TrainingArguments): Converts internal train arguments to
            HuggingFace TrainingArguments format.
    Raises:
        AssertionError: If task type is inconsistent with model and dataset types.
    """

    task: TaskName
    train: TrainArguments
    model: BaseModelArguments
    dataset: BaseDataArguments
    logger: WandbArguments

    resume_checkpoint_path: Optional[str | bool] = False
    layers_to_train: list[str] = field(default_factory=list)

    @property
    def hf_train(self) -> TrainingArguments:
        """Return Hugging Face training arguments."""
        return TrainingArguments(**vars(self.train))

    def _check_task_consistency(self):
        # Validate consistency task <-> (model, dataset)
        if self.task is TaskName.NLP:
            req_types = (NLPModelArguments, NLPDataArguments)
        else:  # guaranteed to be 'vision'
            req_types = (VisionModelArguments, VisionDataArguments)

        to_check = (self.model, self.dataset)
        msg = (
            f"Task '{self.task}' requires type(model)=NLPModelArguments "
            "and type(dataset)=NLPDataArguments, got: "
            f"type(model)={to_check[0].__class__.__name__} and "
            f"type(dataset)={to_check[1].__class__.__name__}"
        )

        assert all(isinstance(var, req_t) for var, req_t in zip(to_check, req_types)), msg

    def __post_init__(self):
        self._check_task_consistency()


cs = ConfigStore.instance()
cs.store(name="base_config", node=FinetuneConfig)
cs.store(group="train", name="base_train", node=TrainArguments)

# For vision task
cs.store(group="model", name="nlp_model", node=NLPModelArguments)
cs.store(group="dataset", name="nlp_dataset", node=NLPDataArguments)

# For nlp task
cs.store(group="model", name="vision_model", node=VisionModelArguments)
cs.store(group="dataset", name="vision_dataset", node=VisionDataArguments)

cs.store(group="logger", name="base_logger", node=WandbArguments)


@parse_structured_config(version_base=None, config_path="../config", config_name="config")
def main(cfg: FinetuneConfig | None = None) -> None:
    """Print the resolved finetuning config for debugging."""
    assert cfg is not None
    assert isinstance(cfg, FinetuneConfig), (
        "Something wrong in the conversion from OmegaConf.DictConfig to Config"
    )
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
