"""Shared CPU-only fixtures for FlexRank package tests."""

# pylint: disable=missing-function-docstring

import pytest
import torch

from flexrank.layers.conv import ODConv2d
from flexrank.layers.linear import ODLinear


class TinyODModel(torch.nn.Module):
    """Small model with both linear and convolutional OD layers."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = ODLinear(4, 3, bias=True)
        self.conv = ODConv2d(2, 3, kernel_size=3, padding=1, bias=True)


@pytest.fixture(autouse=True)
def _deterministic_torch() -> None:
    torch.manual_seed(0)


@pytest.fixture
def tiny_od_model() -> TinyODModel:
    return TinyODModel()
