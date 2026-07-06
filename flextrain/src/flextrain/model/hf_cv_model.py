"""Utilities to load a CV model from HuggingFace transformers library."""

from functools import partial
import numpy as np
import evaluate
from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoConfig
from transformers.trainer_utils import EvalPrediction
from flexrank.utils import init_logger

log = init_logger(__name__)


def load_cv_model_from_hf(
    model_name_or_path: str,
    torch_dtype,
    num_labels: int,
    load_pretrained: bool = False,
    cache_dir: str = None,
    trust_remote_code: bool = True,
    use_auth_token: bool = True,
):
    """
    Load a computer vision model and image processor from Hugging Face Hub.
    This function loads an image classification model either from pretrained weights
    or from a configuration file, along with its corresponding image processor.
    Args:
        model_name_or_path (str): The model identifier or path on Hugging Face Hub.
        torch_dtype: The PyTorch data type to use for the model.
        num_labels (int): The number of classification labels for the model.
        load_pretrained (bool, optional): If True, load pretrained weights. If False, initialize
            from config only. Defaults to False.
        cache_dir (str, optional): Directory to cache downloaded models and processors.
            Defaults to None.
        trust_remote_code (bool, optional): Whether to trust and execute custom code from
            the remote repository. Defaults to True.
        use_auth_token (bool, optional): Whether to use authentication token for private models.
            Defaults to True.
    Returns:
        tuple: A tuple containing:
            - model (AutoModelForImageClassification): The loaded image classification model.
            - processor (AutoImageProcessor): The image processor for the model.
    Raises:
        Exceptions from transformers library if model loading fails.
    """
    log.info("loading base model %s...", model_name_or_path)

    if load_pretrained:
        log.info("loading pretrained weights")
        model_fn = partial(
            AutoModelForImageClassification.from_pretrained,
            model_name_or_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            token=use_auth_token,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            trust_remote_code=trust_remote_code,
            token=use_auth_token,
        )
        model_fn = partial(AutoModelForImageClassification.from_config, config)

    part_model = partial(model_fn, dtype=torch_dtype)
    model = part_model()

    processor = AutoImageProcessor.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        token=use_auth_token,
    )

    return model, processor


def get_cv_metrics(eval_pred: EvalPrediction) -> dict:
    """Loads the `accuracy` metric for image classification models"""
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)
