"""Utilities for loading datasets and deriving train/eval/calibration splits."""

from typing import Any, Callable, NamedTuple, Optional, TypeAlias, Union, cast

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import DefaultDataCollator, DataCollatorForLanguageModeling

from flexrank.utils import init_logger
from flextrain.utils.args import NLPDataArguments, TaskName, VisionDataArguments

from .hf_cv_data import get_cv_dataset
from .hf_nlp_data import get_nlp_dataset

__all__ = [
    "load_dataset_from_hf",
    "get_eval_calib_splits",
    "get_train_eval_calib_splits",
    "DatasetSplits",
    "DatasetSplitsWithCollator",
]

DatasetType: TypeAlias = Union[Dataset, IterableDataset]
DatasetDictType: TypeAlias = Union[DatasetDict, IterableDatasetDict]
CollatorType: TypeAlias = Union[
    DefaultDataCollator,
    DataCollatorForLanguageModeling,
    Callable[[list[Any]], dict[str, Any]],
]


class DatasetSplits(NamedTuple):
    """Immutable container for train, validation, and calibration splits."""

    train: DatasetType
    val: DatasetType
    calib: DatasetType


class DatasetSplitsWithCollator(NamedTuple):
    """Dataset splits bundled with the collator used to batch them."""

    train: DatasetType
    val: DatasetType
    calib: DatasetType
    collator: CollatorType


log = init_logger(__name__)


def load_dataset_from_hf(
    task: TaskName,
    data_args: NLPDataArguments | VisionDataArguments,
    processor,
    data_seed: Optional[int] = None,
) -> DatasetSplitsWithCollator:
    """
    Load a dataset from Hugging Face and derive the splits used by training.

    Args:
        task (TaskName): The type of task (NLP or Vision).
        data_args (NLPDataArguments | VisionDataArguments): Data configuration
            arguments for the selected task.
        processor: The tokenizer (for NLP) or processor (for Vision) to use for
            preprocessing.
        data_seed (Optional[int]): Random seed used when shuffling vision data.

    Returns:
        DatasetSplitsWithCollator: Training, evaluation, calibration splits and
            the collate function used to batch them.
    """
    if task is TaskName.NLP:
        args = cast(NLPDataArguments, data_args)
        dataset, collate_fn = get_nlp_dataset(
            path=args.path,
            name=args.name,
            tokenizer=processor,
            streaming=args.streaming,
            sequence_length=args.sequence_length,
            cache_file_names=args.cache_file_names,
        )
    else:
        args = cast(VisionDataArguments, data_args)
        dataset, collate_fn = get_cv_dataset(
            path=args.path,
            name=args.name,
            processor=processor,
            streaming=args.streaming,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            cache_file_names=args.cache_file_names,
            image_col=args.image_col,
            label_col=args.label_col,
        )
        # Shuffle the dataset since label-ordered samples can skew extracted subsets.
        dataset = dataset.shuffle(seed=data_seed)

    datasets = get_train_eval_calib_splits(
        dataset,
        args.eval_ds_size,
        args.calib_ds_size,
    )
    return DatasetSplitsWithCollator(
        datasets.train,
        datasets.val,
        datasets.calib,
        collate_fn,
    )


def get_eval_calib_splits(
    dataset: DatasetDictType,
    eval_ds_size: int,
    calib_ds_size: int,
    verbose: bool = False,
) -> tuple[DatasetType, DatasetType]:
    """
    Extract evaluation and calibration splits from a dataset dictionary.

    This function retrieves a contiguous subset of samples from an available
    split (`eval`, `validation`, `test`, or `train`) and divides it into
    evaluation and calibration subsets.

    Args:
        dataset: A dictionary-like object containing named dataset splits.
        eval_ds_size: Number of samples to allocate for the evaluation split.
        calib_ds_size: Number of samples to allocate for the calibration split.
        verbose: If True, log information about the extraction process.

    Returns:
        tuple[DatasetType, DatasetType]: The evaluation and calibration splits.

    Raises:
        ValueError: If the dataset type is neither Dataset nor IterableDataset.
        AssertionError: If either requested split size is not greater than zero.
        StopIteration: If none of the expected split keys are present.
    """

    def get_contiguous_subset(
        ds: DatasetType,
        subset_size: int,
        remove_from_src: bool,
    ) -> tuple[DatasetType, DatasetType]:
        if isinstance(ds, Dataset):
            subset = ds.select(range(subset_size))
            if remove_from_src:
                ds = ds.select(range(subset_size, len(ds)))
        elif isinstance(ds, IterableDataset):
            subset = ds.take(subset_size)
            if remove_from_src:
                ds = ds.skip(subset_size)
        else:
            raise ValueError(f"Unhandled dataset type {type(ds)}")

        return subset, ds

    assert all((eval_ds_size > 0, calib_ds_size > 0)), (
        "Both splits size must be > 0, "
        f"got eval_ds_size={eval_ds_size}, calib_ds_size={calib_ds_size}"
    )
    eval_key = next(key for key in ("eval", "validation", "test", "train") if key in dataset)
    subset_size = eval_ds_size + calib_ds_size

    # If we selected from train or test, remove those samples from the source split.
    take_from_split = eval_key in ("test", "train")
    eval_calib_split, dataset[eval_key] = get_contiguous_subset(
        dataset[eval_key],
        subset_size,
        remove_from_src=take_from_split,
    )
    eval_ds, calib_ds = get_contiguous_subset(
        eval_calib_split,
        eval_ds_size,
        remove_from_src=True,
    )

    if verbose:
        log.info(
            "Extracting %s samples from %s split: %s for validation and %s for calibration",
            subset_size,
            eval_key,
            eval_ds_size,
            calib_ds_size,
        )

    return eval_ds, calib_ds


def get_train_eval_calib_splits(
    dataset: DatasetDictType,
    eval_ds_size: int,
    calib_ds_size: int,
    verbose: bool = False,
) -> DatasetSplits:
    """
    Split a dataset into training, evaluation, and calibration subsets.

    Args:
        dataset: The input dataset dictionary containing at least a `train` split.
        eval_ds_size: The number of samples to allocate for the evaluation split.
        calib_ds_size: The number of samples to allocate for the calibration split.
        verbose: If True, log additional information during the split.

    Returns:
        DatasetSplits: The training, evaluation, and calibration datasets.
    """
    eval_ds, calib_ds = get_eval_calib_splits(
        dataset,
        eval_ds_size,
        calib_ds_size,
        verbose,
    )
    return DatasetSplits(dataset["train"], eval_ds, calib_ds)
