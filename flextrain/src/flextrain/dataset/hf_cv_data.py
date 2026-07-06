"""Computer-vision dataset loading and preprocessing helpers."""

from io import BytesIO
from typing import Optional

from datasets import DatasetDict, IterableDatasetDict, Dataset, IterableDataset, load_dataset, Image
from PIL import Image as PILImage
from torch import nn
from torchvision.transforms import v2
from transformers import DefaultDataCollator
from transformers import AutoImageProcessor

from flexrank.utils import init_logger

log = init_logger(__name__)

DatasetDictType = DatasetDict | IterableDatasetDict
DatasetType = Dataset | IterableDataset


def _robust_load_transform(batch, transform, image_col: str, label_col: str):
    # Handles a bug od HF datasets regarding error handling for bad images.
    # This occurs only in non-streaming mode.
    # 1. Issue documented: https://github.com/huggingface/datasets/issues/7632
    # 2. Request graceful error handling: https://github.com/huggingface/datasets/issues/7632
    # 3. PR with solution: https://github.com/huggingface/datasets/pull/7638
    # Up to date (07/01/2026) the PR hasn't been merged yet, hence this workaround
    images, labels = [], []
    for raw_image, label in zip(batch[image_col], batch[label_col]):
        try:
            image = PILImage.open(BytesIO(raw_image["bytes"])).convert("RGB")
            image = transform(image)
            images.append(image)
            labels.append(label)
        except (KeyError, OSError, ValueError) as exc:
            log.warning("Skipping unreadable image: %s", exc)
            continue
    assert len(images) > 0
    return images, labels


def _preprocess_function(examples, processor, transform, image_col: str, label_col: str):
    if isinstance(examples[image_col], list):
        # this happens in non-streaming mode
        imgs, labels = _robust_load_transform(examples, transform, image_col, label_col)
        pixel_values = processor(imgs, return_tensors="pt")["pixel_values"]
        labels = [int(y) for y in labels]
        return {"pixel_values": pixel_values, "labels": labels}

    imgs, labels = examples[image_col], examples[label_col]
    imgs = transform(imgs)
    pixel_values = processor(imgs, return_tensors="pt")["pixel_values"][0]
    return {"pixel_values": pixel_values, "labels": int(labels)}


def _apply_map_like(ds, fn, *, fn_kwargs: dict, remove_columns=None):
    """Behave like map() on both Dataset and IterableDataset."""
    if isinstance(ds, IterableDataset):
        return ds.map(fn, fn_kwargs=fn_kwargs, remove_columns=remove_columns)
    # with_transform ignores remove_columns, but returns only fn(...) output anyway.
    return ds.with_transform(lambda ex: fn(ex, **fn_kwargs))


def _preprocess_dataset(
    dataset: DatasetDictType, processor, train_tfm, eval_tfm, image_col: str, label_col: str
):
    for name, ds in dataset.items():
        transform = train_tfm if name == "train" else eval_tfm
        dataset[name] = _apply_map_like(
            ds,
            _preprocess_function,
            fn_kwargs={
                "processor": processor,
                "transform": transform,
                "image_col": image_col,
                "label_col": label_col,
            },
            remove_columns=[image_col, label_col],
        )
    return dataset


def _load_dataset_wrapper(
    dataset_name: str,
    *args,
    image_col: str = "image",
    label_col: str = "label",
    **kwargs,
):

    dataset = load_dataset(dataset_name, *args, **kwargs)
    if not kwargs.get("streaming", False):
        dataset = dataset.cast_column(image_col, Image(decode=False))
    # Get columns from a split (works for both DatasetDict and IterableDatasetDict)
    first_split = next(iter(dataset))
    all_columns = set(dataset[first_split].features.keys())
    columns_toremove = list(all_columns - {image_col, label_col})

    # Remove columns for each split
    for split in dataset.keys():
        dataset[split] = dataset[split].remove_columns(columns_toremove)

    return dataset


def _get_transforms(
    processor: AutoImageProcessor, resize_size: int, crop_size: int
) -> tuple[nn.Module, nn.Module]:
    # Override transformations in the processor, leave only normalization
    processor.size = {"height": crop_size, "width": crop_size}
    processor.do_resize = False
    processor.do_center_crop = False

    # Train and test trasformations
    # See dinov3/data/transforms.py in facebookresearch/dinov3 for the
    # reference training transform.
    train_tfm = v2.Compose(
        [
            v2.RandomResizedCrop(crop_size, interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(),
        ]
    )
    # See the same file for the reference eval transform.
    eval_tfm = v2.Compose([v2.Resize(resize_size), v2.CenterCrop(crop_size)])

    return train_tfm, eval_tfm


def get_cv_dataset(
    path: str,
    name: str,
    processor: AutoImageProcessor,
    streaming: bool,
    resize_size: int,
    crop_size: int,
    split: Optional[str] = None,
    cache_file_names: Optional[str] = None,
    image_col: str = "image",
    label_col: str = "label",
):
    """
    Load and preprocess a computer vision dataset from Hugging Face datasets.
    Applies appropriate image transformations for training and evaluation phases.
    Args:
        path (str): Path or identifier of the dataset to load.
        name (str): Unused, kept for consistency with LM dataset loader.
        processor (AutoImageProcessor): Image processor for preprocessing images.
        streaming (bool): Whether to stream the dataset or download it entirely.
        image_size (int): Target size for image resizing and cropping.
        split (Optional[str], optional): Unused, kept for consistency with LM
            dataset loader. Defaults to None.
        cache_file_names (Optional[str], optional): Unused, kept for consistency
            with LM dataset loader. Defaults to None.
        image_col (str, optional): Name of the image column in the dataset. Defaults to "image".
        label_col (str, optional): Name of the label column in the dataset. Defaults to "label".
    Returns:
        tuple: A tuple containing:
            - ds: Preprocessed dataset with applied transformations.
            - collate_fn: Data collator function for batching samples during training.
    """
    del name, split, cache_file_names
    train_tfm, eval_tfm = _get_transforms(processor, resize_size, crop_size)
    ds = _load_dataset_wrapper(path, streaming=streaming, image_col=image_col, label_col=label_col)
    ds = _preprocess_dataset(ds, processor, train_tfm, eval_tfm, image_col, label_col)

    collate_fn = DefaultDataCollator()

    return ds, collate_fn
