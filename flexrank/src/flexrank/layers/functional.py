"""Low-level tensor helpers for decomposed linear and convolution layers."""

from typing import Optional

import torch
from torch.nn.functional import conv2d, linear

####################
### LINEAR UTILS ###
####################


def linear_slice_in(weight: torch.nn.Parameter, inner_dim: int) -> torch.Tensor:
    """Return the active input-factor rows for a given inner dimension."""
    return weight[:inner_dim, :]


def linear_slice_out(weight: torch.nn.Parameter, inner_dim: int) -> torch.Tensor:
    """Return the active output-factor columns for a given inner dimension."""
    return weight[:, :inner_dim]


def od_sliced_linear(
    inputs: torch.Tensor,
    weight_in: torch.nn.Parameter,
    weight_out: torch.nn.Parameter,
    inner_dim: int,
    bias: Optional[torch.nn.Parameter],
    rank_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply the low-rank linear transform using sliced factor weights."""
    # pylint: disable=not-callable
    del rank_mask
    slice_in = linear_slice_in(weight_in, inner_dim)
    slice_out = linear_slice_out(weight_out, inner_dim)

    out = linear(inputs, slice_in)
    return linear(out, slice_out, bias)


def od_dense_linear(
    inputs: torch.Tensor,
    weight_in: torch.nn.Parameter,
    weight_out: torch.nn.Parameter,
    inner_dim: int,
    bias: Optional[torch.nn.Parameter],
    rank_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the low-rank linear transform by masking inactive channels."""
    # pylint: disable=not-callable
    del inner_dim

    intermediate = linear(inputs, weight_in)
    if rank_mask.dtype != intermediate.dtype:
        rank_mask = rank_mask.to(dtype=intermediate.dtype)
    intermediate = intermediate * rank_mask.view(
        *([1] * (intermediate.dim() - 1)),
        -1,
    )
    return linear(intermediate, weight_out, bias)


####################
### CONV2D UTILS ###
####################


def conv2d_slice_in(weight: torch.nn.Parameter, inner_dim: int) -> torch.Tensor:
    """Return the active input-factor filters for a given inner dimension."""
    return weight[:inner_dim, :, :, :]


def conv2d_slice_out(weight: torch.nn.Parameter, inner_dim: int) -> torch.Tensor:
    """Return the active output-factor filters for a given inner dimension."""
    return weight[:, :inner_dim, :, :]


# Mirrors `torch.nn.functional.conv2d` arguments plus `inner_dim`.
# pylint: disable=too-many-arguments,too-many-positional-arguments
def od_sliced_conv2d(
    inputs: torch.Tensor,
    weight_in: torch.nn.Parameter,
    weight_out: torch.nn.Parameter,
    inner_dim: int,
    bias: Optional[torch.nn.Parameter],
    stride: int,
    padding: int,
    dilation: int,
    rank_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply the low-rank convolution using sliced factor filters."""
    # pylint: disable=not-callable
    del rank_mask
    slice_in = conv2d_slice_in(weight_in, inner_dim)
    slice_out = conv2d_slice_out(weight_out, inner_dim)

    out = conv2d(inputs, slice_in, None, stride, padding, dilation, groups=1)
    return conv2d(out, slice_out, bias, 1, 0, dilation=1, groups=1)


# Mirrors `torch.nn.functional.conv2d` arguments plus `inner_dim`.
# pylint: disable=too-many-arguments,too-many-positional-arguments
def od_dense_conv2d(
    inputs: torch.Tensor,
    weight_in: torch.nn.Parameter,
    weight_out: torch.nn.Parameter,
    inner_dim: int,
    bias: Optional[torch.nn.Parameter],
    stride: int,
    padding: int,
    dilation: int,
    rank_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the low-rank convolution by masking inactive channels."""
    # pylint: disable=not-callable
    del inner_dim

    out = conv2d(inputs, weight_in, None, stride, padding, dilation, groups=1)
    if rank_mask.dtype != out.dtype:
        rank_mask = rank_mask.to(dtype=out.dtype)
    out = out * rank_mask.view(1, -1, 1, 1)

    return conv2d(out, weight_out, bias, 1, 0, dilation=1, groups=1)
