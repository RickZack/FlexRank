"""Decomposition type markers."""

from enum import StrEnum, auto

__all__ = ["SVDType"]


class SVDType(StrEnum):
    """Supported SVD decomposition modes."""

    SVD = auto()
    DATA_SVD = auto()
