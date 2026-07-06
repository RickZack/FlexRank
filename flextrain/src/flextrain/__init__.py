"""Top-level package for flextrain.

Marks the directory as a Python package and exposes common subpackages.
"""

__version__ = "0.0.1"

from . import callbacks, config, dataset, model, utils

__all__ = [
    "callbacks",
    "config",
    "dataset",
    "model",
    "utils",
]
