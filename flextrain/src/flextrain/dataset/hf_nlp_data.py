"""NLP dataset loading and tokenization helpers."""

from typing import Optional

from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset, IterableDataset
from transformers import DataCollatorForLanguageModeling

from flexrank.utils import init_logger

log = init_logger(__name__)

__all__ = ["get_nlp_dataset"]

DatasetDictType = DatasetDict | IterableDatasetDict
DatasetType = Dataset | IterableDataset


def _tokenize_function(examples, tokenizer, sequence_length):
    return tokenizer(
        text=examples["text"],
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
    )


def _load_dataset_wrapper(dataset_name: str, cgf: str, *args, **kwargs):
    dataset = load_dataset(dataset_name, cgf, *args, **kwargs)

    # Get columns from a split (works for both DatasetDict and IterableDatasetDict)
    first_split = next(iter(dataset))
    all_columns = set(dataset[first_split].features.keys())
    columns_toremove = list(all_columns - {"text"})

    # Remove columns for each split
    for split in dataset.keys():
        dataset[split] = dataset[split].remove_columns(columns_toremove)

    return dataset


def _tokenize_dataset(
    dataset: DatasetDictType,
    tokenizer,
    sequence_length: int,
    cache_file_names: Optional[dict[str, str]] = None,
):
    map_kwargs = {}

    if cache_file_names:
        if isinstance(dataset, DatasetDict):
            map_kwargs["cache_file_names"] = cache_file_names
        else:  # dataset is IterableDatasetDict
            log.warning(
                "A cache file path has been provided, but the dataset is in "
                "streaming mode. Ignoring the cache file."
            )

    return dataset.map(
        _tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "sequence_length": sequence_length},
        remove_columns=["text"],
        **map_kwargs,
    )


def get_nlp_dataset(
    path: str,
    name: str,
    tokenizer,
    streaming: bool,
    sequence_length: int = 1024,
    split: str = None,
    cache_file_names: Optional[dict[str, str]] = None,
):
    """
    Load and prepare an NLP dataset for causal language modeling tasks.

    This function loads a dataset from the Hugging Face Hub or local path,
    tokenizes it, and returns a prepared dataset along with a collate function
    for training causal language models.

    Args:
        path (str): The path or identifier of the dataset.
        name (str): The specific configuration or subset name of the dataset.
        tokenizer: The tokenizer instance to use for encoding text data.
        streaming (bool): Whether to stream the dataset or download and map it.
        sequence_length (int, optional): Maximum tokenized sequence length.
            Defaults to 1024.
        split (str, optional): Dataset split to load. Defaults to None.
        cache_file_names (dict[str, str], optional): Arrow cache files for
            tokenized datasets. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - dataset: The tokenized and prepared dataset.
            - collate_fn: Causal language modeling collator.
    """
    dataset = _load_dataset_wrapper(path, name, split=split, streaming=streaming)
    dataset = _tokenize_dataset(
        dataset, tokenizer, sequence_length, cache_file_names=cache_file_names
    )

    collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # For causal LM (GPT-2/GPT-Neo)
    )

    return dataset, collate_fn
