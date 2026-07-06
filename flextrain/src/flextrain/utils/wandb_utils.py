"""Weights & Biases setup and resume helpers."""

from dataclasses import asdict, is_dataclass
import json
import os
from typing import Any, Mapping, Optional

import wandb
from accelerate import Accelerator
from flexrank.utils import init_logger

from .args import SerializableMixin, WandbArguments


log = init_logger(__name__)
accelerator = Accelerator()


def _coerce_config_to_dict(config: Any) -> dict:
    if isinstance(config, Mapping):
        data = dict(config)
    elif hasattr(config, "to_dict") and callable(getattr(config, "to_dict")):
        data = config.to_dict()
    elif is_dataclass(config):
        data = asdict(config)
    elif hasattr(config, "__dict__"):
        data = dict(vars(config))
    else:
        data = {"value": config}
    return SerializableMixin.serialize_value(data)


def _strip_keys(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {k: _strip_keys(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [_strip_keys(v, keys) for v in value]
    return value


def _load_api_key_from_file(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("WANDB_API_KEY")
    except (OSError, json.JSONDecodeError) as exc:
        log.error("Unable to load API key from %s: %s", file_path, exc)
        return None


def _load_wandb_config(path: str, args: WandbArguments) -> None:
    try:
        with open(os.path.join(path, "wandb.json"), "r", encoding="utf-8") as f:
            run_wandb_json: dict = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.error("Unable to load previous WandB configuration: %s", exc)
        return

    args.id = run_wandb_json.get("id")
    args.resume = "must"


def _save_wandb_config(path: str, run: wandb.sdk.wandb_run.Run) -> None:
    run_wandb_json = {
        "id": run.id,
        "project": run.project,
        "entity": run.entity,
        "config": run.config.as_dict(),
    }
    with open(os.path.join(path, "wandb.json"), "w", encoding="utf-8") as f:
        json.dump(run_wandb_json, f, indent=2)


@accelerator.on_main_process
def setup_wandb(
    *,
    wandb_args: WandbArguments,
    run_config: Any,
    output_dir: str,
    load_path: Optional[str] = None,
) -> wandb.sdk.wandb_run.Run:
    """
    Initialize W&B logging with a config dict that excludes secrets.

    Args:
        wandb_args: W&B runtime arguments (may include api_key_file path).
        run_config: Arbitrary config object or dict to log.
        output_dir: Directory where wandb.json will be saved.
        load_path: Optional path to resume a previous run.
    """
    if wandb_args.api_key_file:
        api_key = _load_api_key_from_file(wandb_args.api_key_file)
        if api_key:
            wandb.login(key=api_key)

    os.makedirs(output_dir, exist_ok=True)
    if load_path:
        _load_wandb_config(load_path, wandb_args)

    wandb_kwargs = {k: v for k, v in vars(wandb_args).items() if k != "api_key_file"}
    config_dict = _strip_keys(_coerce_config_to_dict(run_config), {"api_key_file"})

    run = wandb.init(**wandb_kwargs, config=config_dict)
    _save_wandb_config(output_dir, run)
    return run
