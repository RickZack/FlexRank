"""Serialization helpers for FlexRank experiment data."""

import os
import pickle
from dataclasses import dataclass, asdict, field, is_dataclass
from typing import Optional, Union, TypeAlias

from accelerate import Accelerator
from datasets import Dataset, IterableDataset
from transformers.trainer_utils import get_last_checkpoint

from flexrank.profiles import (
    MergedProfilesData,
    ProfileAlgoSolution,
    ProfilesData,
    get_sol_from_dict,
)
from flexrank.utils import init_logger
from flextrain.model import FlexRankModel

from .args import DecompositionArguments
from .distillation_trainer import DistillationTrainer

DatasetType: TypeAlias = Union[Dataset, IterableDataset]

log = init_logger(__name__)
accelerator = Accelerator()


@dataclass
class FlexRankData:
    """Container for FlexRank experiment metadata and checkpoints.

    Attributes:
        trainer (DistillationTrainer): Trainer instance (not serialized).
        profiles_data (MergedProfilesData): Data about submodel profiles.
        full_model_parameters (int): Parameter count of the original model.
        full_model_loss (float): Baseline loss of the original model.
        decomp_args (DecompositionArguments): Decomposition configuration used.
        end_profiles_loss (Optional[list[float]]): Final per-profile losses when available.

    Notes:
        - trainer is held but not serialized; a trainer must be supplied when loading.
        - save() writes "flexrank_data.pkl" and uses the trainer to store a model checkpoint.
        - load() restores model weights from the last checkpoint.
    """

    trainer: Optional[DistillationTrainer]
    profiles_data: MergedProfilesData
    sol: ProfileAlgoSolution
    full_model_parameters: int
    full_model_loss: float
    decomp_args: DecompositionArguments
    end_profiles_loss: Optional[list[float]] = None
    eval_history: dict[int, dict] = field(default_factory=dict)
    last_profiles_data: Optional[ProfilesData] = None

    def _convert_to_dict(self):
        """Return serializable dict of fields (dataclasses -> dict), excluding 'trainer'."""
        d = {
            k: v if not is_dataclass(v) else asdict(v)
            for k, v in vars(self).items()
            if k != "trainer"
        }
        d.update({"sol_cls": self.sol.__class__.__name__})
        return d

    @staticmethod
    @accelerator.on_main_process
    def _store_fields_to_pickle(data: dict, output_dir: str):
        """
        Write a Python dictionary to "flexrank_data.pkl" inside the given output
        directory.
        If the file already exists it will be overwritten.

        Args:
            data (dict): The dictionary to serialize. Keys and values must be JSON-serializable.
            output_dir (str): Path to the directory where "flexrank_data.pkl" will be created.

        Returns:
            None

        Raises:
            FileNotFoundError: If `output_dir` does not exist.
            PermissionError: If the process lacks permission to create or write the file.
            OSError: For other I/O related errors during file creation or writing.
        """
        with open(os.path.join(output_dir, "flexrank_data.pkl"), "wb") as f:
            pickle.dump(data, f)

    def save(self, output_dir: Optional[str] = None):
        """
        Save the object's state serialized data to a pickle file.

        Args:
            output_dir (Optional[str]):
                Path to the directory where the representation of this object should
                be written. If None, the method uses self.trainer.args.output_dir.

        Returns:
            None
        """
        out_dir = output_dir or self.trainer.args.output_dir
        FlexRankData._store_fields_to_pickle(self._convert_to_dict(), out_dir)

    @staticmethod
    def _load_fields_from_pickle(input_dir: str) -> dict:
        """
        Load FlexRank data fields from a .pkl file.

        Args:
            input_dir (str): The directory path containing the 'flexrank_data.json' file.

        Returns:
            dict: A dictionary containing the converted FlexRank data fields, with dataclasses
                  reconstructed from their data fields.

        Raises:
            FileNotFoundError: If 'flexrank_data.pkl' does not exist in the specified directory.
        """
        with open(os.path.join(input_dir, "flexrank_data.pkl"), "rb") as f:
            return FlexRankData._convert_dataclasses_back(pickle.load(f))

    @staticmethod
    def _convert_dataclasses_back(data: dict) -> dict:
        data["sol"] = get_sol_from_dict(data["sol_cls"], data["sol"])
        data["profiles_data"] = MergedProfilesData(**data["profiles_data"])
        data["decomp_args"].pop("calib_ds_size", None)
        data["decomp_args"] = DecompositionArguments(**data["decomp_args"])
        del data["sol_cls"]

        return data

    @staticmethod
    def load(
        trainer: Optional[DistillationTrainer] = None,
        input_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> "FlexRankData":
        """
        Load FlexRank data from a previous checkpoint directory.

        This method restores FlexRank data from disk, including all serialized fields
        and the associated trainer's model state. Model weights are restored from the latest
        checkpoint found in the input directory.

        Args:
            trainer (Optional[DistillationTrainer]): Trainer instance to attach
                to the loaded data. Defaults to None.
            input_dir (Optional[str]): The directory path containing the saved FlexRank data
                and checkpoint files. Defaults to None.

        Returns:
            FlexRankData: A FlexRankData instance populated with loaded data and the
                provided trainer.

        Note:
            This method only restores the model weights. To fully restore from a checkpoint,
            pass `reload_from_checkpoint` parameter to trainer.train().
        """
        if verbose:
            log.info("Loading previous FlexRank data from %s", input_dir)
        data = FlexRankData._load_fields_from_pickle(input_dir)
        data["trainer"] = trainer

        # We load only the model, if one really want to restore from a checkpoint,
        # pass `reload_from_checkpoint` to trainer.train()
        if trainer is not None:
            checkpoint_path = get_last_checkpoint(input_dir)
            loaded = FlexRankModel.from_pretrained(checkpoint_path)
            trainer.model.load_state_dict(loaded.state_dict())
            del loaded

        return FlexRankData(**data)
