"""Utilities for flexrank.trainers"""

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch

from ..utils.logger import init_logger

__all__ = ["save_results", "load_results", "save_model", "load_model"]

logger = init_logger(__name__)


def _to_jsonable(obj: Any):
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "_asdict"):
        return {k: _to_jsonable(v) for k, v in obj._asdict().items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, RuntimeError):
            pass
    return obj


def save_results(output_dir: str, name: str, res: Any):
    """Save results to a JSON file in the specified output directory."""
    path = Path(output_dir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_to_jsonable(res), f, indent=4)
    logger.info("Results saved to %s", path)


def load_results(output_dir: str, name: str, res_type: type | None = None) -> dict[str, Any] | Any:
    """Load cached results and optionally cast them to a result type."""
    path = Path(output_dir) / f"{name}.json"
    with path.open("r") as f:
        data = json.load(f)
    if res_type is None:
        return data
    return res_type(**data)


def save_model(model: torch.nn.Module, output_dir: str, name: str = "model.pth"):
    """Save a PyTorch model's state_dict to the specified output directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / name
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to %s", model_path)


def load_model(model: torch.nn.Module, output_dir: str, name: str = "model.pth"):
    """Load a PyTorch model's state_dict from the specified output directory."""
    model_path = Path(output_dir) / name
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    device = next(model.parameters()).device
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    logger.info("Model loaded from %s", model_path)
