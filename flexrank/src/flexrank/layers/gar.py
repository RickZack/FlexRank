"""Utilities implementing the GAR format."""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from flexrank.utils.logger import init_logger

log = init_logger(__name__)


class GARForm(NamedTuple):
    """
    Compact GAR representation of a rank-``r`` factorization.

    If ``UV^T = tilde_U tilde_V^T`` with

    ``tilde_U = [I_r; hat_U]``,

    then this object stores:
    - ``u``: the non-trivial block ``hat_U`` with shape ``(m - r, r)``
    - ``v``: the transformed right factor ``tilde_V`` with shape ``(n, r)``
    """

    u: Tensor
    v: Tensor


class GARResult(NamedTuple):
    """
    Output of ``make_gar``.

    Attributes:
        gar: Compact GAR form ``(hat_U, tilde_V)``.
        g: Gauge matrix ``G = U_{[:r]}^{-1}``.
    """

    gar: GARForm
    g: Tensor


def _prepare_gar_inputs(u: Tensor, v: Tensor, r: int) -> tuple[Tensor, Tensor, torch.dtype]:
    """Validate inputs and upcast tensors for stable GAR construction."""
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError(f"`u` and `v` must be matrices, got shapes {u.shape} and {v.shape}.")
    if u.shape[1] != v.shape[1]:
        raise ValueError(
            f"`u` and `v` must share the same inner dimension, got {u.shape[1]} and {v.shape[1]}."
        )
    if r <= 0:
        raise ValueError(f"`r` must be positive, got {r}.")
    if r > u.shape[1]:
        raise ValueError(f"`r` must be <= the inner dimension {u.shape[1]}, got {r}.")
    if r > u.shape[0]:
        raise ValueError(f"`r` must be <= the number of rows in `u` ({u.shape[0]}), got {r}.")

    original_dtype = u.dtype
    work_dtype = torch.float64
    return u.to(work_dtype), v.to(work_dtype), original_dtype


def _finalize_gar_factors(
    u_r: Tensor,
    v_r: Tensor,
    pivot: Tensor,
    g: Tensor,
    output_dtypes: tuple[torch.dtype, torch.dtype],
) -> tuple[Tensor, Tensor]:
    """Build compact GAR factors and validate the normalized pivot block."""
    u_dtype, v_dtype = output_dtypes
    eye = torch.eye(pivot.shape[0], dtype=u_r.dtype, device=u_r.device)
    hat_u = torch.linalg.solve(  # pylint: disable=not-callable
        pivot.T,
        u_r[pivot.shape[0] :, :].T,
    ).T
    u_tilde_top = pivot @ g

    tol_dtype = u_dtype if u_dtype.is_floating_point else u_tilde_top.dtype
    tol = 10 * torch.finfo(tol_dtype).eps if u_tilde_top.dtype.is_floating_point else 0.0
    if not torch.allclose(u_tilde_top, eye, atol=tol, rtol=tol):
        raise RuntimeError("GAR gauge construction failed: the pivot block is not the identity.")

    return hat_u.to(u_dtype), (v_r @ pivot.T).to(v_dtype)


def od_gar_linear(
    inputs: Tensor,
    weight_in: Tensor,
    weight_out: Tensor,
    inner_dim: int,
    bias: Tensor | None,
    rank_mask: Tensor | None,
) -> Tensor:
    """
    Apply a GAR-parameterized linear map without materializing the identity block.

    Args:
        inputs: Input tensor with trailing dimension ``in_features``.
        weight_in: ``tilde_V^T`` with shape ``(r, in_features)``.
        weight_out: ``hat_U`` with shape ``(out_features - r, r)``.
        inner_dim: Target rank ``r``.
        bias: Optional bias with shape ``(out_features,)``.
        rank_mask: Unused, accepted to match other OD linear implementations.
    """
    del rank_mask
    latent = F.linear(inputs, weight_in)  # pylint: disable=not-callable

    top_bias = None if bias is None else bias[:inner_dim]
    bottom_bias = None if bias is None else bias[inner_dim:]

    top = latent if top_bias is None else latent + top_bias
    bottom = F.linear(latent, weight_out, bottom_bias)  # pylint: disable=not-callable
    return torch.cat([top, bottom], dim=-1)


def make_gar(u: Tensor, v: Tensor, r: int) -> GARResult:
    """
    Reparametrize ``(U, V)`` into GAR form at target rank ``r``.

    Given the rank-truncated factors ``U = u[:, :r]`` and ``V = v[:, :r]``,
    this function assumes the leading ``r x r`` block of ``U`` is invertible and
    defines the gauge ``G = U_{[:r]}^{-1}``.

    The transformed factors satisfy

    ``U V^T = tilde_U tilde_V^T``,

    with

    ``tilde_U = U G = [I_r; hat_U]``

    and

    ``tilde_V = V G^{-T}``.

    Since the first ``r`` rows of ``tilde_U`` are the identity, only
    ``hat_U`` needs to be stored explicitly.

    Args:
        u: Left factor with shape ``(m, k)``.
        v: Right factor with shape ``(n, k)``.
        r: Target inference rank. Only the first ``r`` columns are used.

    Returns:
        ``GARResult`` containing:
        - ``gar.u``: ``hat_U`` with shape ``(m - r, r)``
        - ``gar.v``: ``tilde_V`` with shape ``(n, r)``
        - ``g``: the gauge matrix ``G`` with shape ``(r, r)``
    """
    u_work, v_work, original_dtype = _prepare_gar_inputs(u, v, r)
    u_r = u_work[:, :r]
    v_r = v_work[:, :r]
    pivot = u_r[:r, :]
    try:
        g = torch.linalg.inv(pivot)  # pylint: disable=not-callable
    except RuntimeError as err:
        log.error(
            "GAR construction failed: the leading %dx%d block is not invertible for rank %d.",
            r,
            r,
            r,
        )
        raise ValueError(
            f"Cannot build a GAR form with rank {r}: the leading {r}x{r} block is not invertible."
        ) from err

    hat_u, tilde_v = _finalize_gar_factors(
        u_r,
        v_r,
        pivot,
        g,
        (original_dtype, v.dtype),
    )

    return GARResult(gar=GARForm(u=hat_u, v=tilde_v), g=g.to(original_dtype))
