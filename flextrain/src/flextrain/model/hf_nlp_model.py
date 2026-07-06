"""Utilities to load an NLP model from HuggingFace transformers library."""

from functools import partial
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from flexrank.utils import init_logger

log = init_logger(__name__)


def load_nlp_model_from_hf(
    model_name_or_path: str,
    torch_dtype,
    load_pretrained: bool = False,
    cache_dir: str = None,
    trust_remote_code: bool = True,
    use_auth_token: bool = True,
):
    """
    Load a causal language model and tokenizer from Hugging Face Model Hub.
    Args:
        model_name_or_path (str): Model identifier or path to pretrained model on Hugging Face Hub.
        torch_dtype: PyTorch data type to load the model weights in.
        load_pretrained (bool, optional): Whether to load pretrained weights. If False, initializes
            model from config only. Defaults to False.
        cache_dir (str, optional): Path to directory where downloaded models are cached.
            Defaults to None (uses Hugging Face default cache).
        trust_remote_code (bool, optional): Whether to allow custom modeling code from the Hub.
            Defaults to True.
        use_auth_token (bool, optional): Whether to use auth token for accessing private models.
            Defaults to True.
    Returns:
        tuple: A tuple containing:
            - model: The loaded AutoModelForCausalLM instance.
            - tokenizer: The corresponding AutoTokenizer with pad_token set to eos_token if needed.
    """
    log.info("loading base model %s...", model_name_or_path)

    if load_pretrained:
        log.info("loading pretrained weights")
        model_fn = partial(
            AutoModelForCausalLM.from_pretrained,
            model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            token=use_auth_token,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            token=use_auth_token,
        )
        model_fn = partial(AutoModelForCausalLM.from_config, config)

    part_model = partial(model_fn, dtype=torch_dtype)
    model = part_model()
    if hasattr(model, "config"):
        # Training never needs autoregressive KV caching. Leaving it enabled makes
        # torch.compile specialize on mutable DynamicCache state in HF causal LMs.
        model.config.use_cache = False

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=False,  # Fast tokenizer giving issues.  # Discrepancy with baseline evaluation
        trust_remote_code=trust_remote_code,
        token=use_auth_token,
    )

    # Set pad_token to eos_token if not already set
    if tokenizer.pad_token is None:
        log.info("Setting pad token to [EOS] token")
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
