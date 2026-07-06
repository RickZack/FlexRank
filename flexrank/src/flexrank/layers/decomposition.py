"""SVD-based decomposition helpers for supported layer types."""
# pylint: disable=invalid-name

from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.linalg import LinAlgError

from flexrank.layers.conv import ODConv2d, build_decomp_conv2d_like
from flexrank.layers.linear import ODLinear, build_decomp_linear_like
from flexrank.types.distributed import DistributedInfo
from flexrank.utils import init_logger

log = init_logger(__name__)


def _rearrage_eigh(L: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rearrange eigenvalues and eigenvectors in descending order.

    This function reorders the eigenvalues and corresponding eigenvectors
    obtained from eigendecomposition so that eigenvalues are sorted from
    largest to smallest.

    Args:
        L (torch.Tensor): A 1D tensor containing eigenvalues.
        Q (torch.Tensor): A 2D tensor where each column is an
            eigenvector corresponding to the eigenvalue at the same index.

    Returns:
        A tuple containing
            - L (torch.Tensor): Eigenvalues sorted in descending order.
            - Q (torch.Tensor): Eigenvectors rearranged to match the sorted eigenvalues.
    """
    idx = torch.arange(L.numel() - 1, -1, -1, device=L.device)
    return L[idx], Q[:, idx]


def _decompose_inputs(inputs_covar: torch.Tensor, device: str, clamp_min: float = 1e-8):
    """
    Diagonalize an input covariance matrix for the data-aware SVD path.

    The covariance is upcast to ``float32`` on the requested device, then
    eigendecomposed and reordered so that the largest eigenvalues come first.
    Small eigenvalues are clamped for numerical stability before taking the
    square root, because the downstream DataSVD path works with the whitened
    input basis rather than the raw covariance directly.

    Decompose the input covariance matrix using eigenvalue decomposition.

    Performs eigenvalue decomposition on the input covariance matrix, rearranges
    the eigenvalues and eigenvectors, applies clamping to ensure numerical
    stability, and returns the square root of eigenvalues with transposed
    eigenvectors.

    Args:
        inputs_covar (torch.Tensor): A square positive semi-definite covariance matrix.
        device (str | torch.device): Device on which to perform computations.
        clamp_min (float, optional): Minimum value for clamping eigenvalues to ensure
            numerical stability. Defaults to 1e-8.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Square root of the clamped eigenvalues.
            - torch.Tensor: Transposed eigenvectors matrix (Q.T).
    """
    inputs_covar = inputs_covar.to(device, torch.float32)
    # pylint: disable=not-callable
    L2, Q = torch.linalg.eigh(inputs_covar)
    L2, Q = _rearrage_eigh(L2, Q)
    L2.clamp_(min=clamp_min)  # matrix is PSD, so clamp in place
    return torch.sqrt(L2), Q.T


class UVDecomp(NamedTuple):
    """Matrix-factor decomposition in `U @ Vt` form."""

    U: torch.Tensor
    Vt: torch.Tensor


class PSQDecomp(NamedTuple):
    """
    SVD-like decomposition stored as ``U @ diag(S) @ Q^T``.

    This is the natural output shape for both the standard SVD path and the
    data-aware variant. The helper ``to_UV_decomp()`` folds ``sqrt(S)`` into
    both sides so the result can be materialized directly into an ``ODLayer``.
    """

    U: torch.Tensor
    S: torch.Tensor
    Qt: torch.Tensor

    # Kept for API compatibility with the rest of the codebase.
    # pylint: disable=invalid-name
    def to_UV_decomp(self, dtype: torch.dtype) -> UVDecomp:
        """
        Convert the singular-value form into a plain two-factor parameterization.

        The singular values are split symmetrically across the left and right
        factors so that ``U_uv @ V_uv^T`` reconstructs the same matrix while
        remaining directly usable as the stored low-rank parameters.

        Args:
            dtype: Target dtype for the returned factors.

        Returns:
            ``UVDecomp`` with both factors cast to ``dtype``.
        """
        u_mat, singular_vals, q_t = self
        sqrt_s = torch.sqrt(singular_vals.reshape(1, -1))
        return UVDecomp((u_mat * sqrt_s).to(dtype), (q_t.T * sqrt_s).to(dtype))


def _compute_datasvd(
    layer_weight: torch.Tensor,
    inputs: torch.Tensor,
    device: str | torch.device,
) -> PSQDecomp:
    """
    Compute the data-dependent singular value decomposition of a layer weight matrix.

    This function performs SVD on a transformed weight matrix that incorporates input
    statistics. It decomposes the inputs, computes the SVD of a transformed weight matrix,
    and normalizes the singular vectors.

    Args:
        layer_weight (torch.Tensor): Layer weight matrix.
        inputs (torch.Tensor): The input data to the layer.
        device (str | torch.device): Device on which to perform computations.

    Returns:
        PSQDecomp: A decomposition of the transformed weight matrix.
    """
    S_X, V_Xt = _decompose_inputs(inputs, device)
    # pylint: disable=not-callable
    U, S, Vh = torch.linalg.svd(torch.diag(S_X) @ V_Xt @ layer_weight.T, full_matrices=False)
    U = V_Xt.T @ torch.diag(S_X ** (-1)) @ U
    norm = torch.linalg.norm(U, dim=0)
    U.div_(norm.unsqueeze(0))
    S.mul_(norm)
    # correct order, since we decomposed W.T
    U, Vh = Vh.T, U.T
    return PSQDecomp(U, S, Vh)


def compute_decomp(
    layer_weight: torch.Tensor,
    device: str | torch.device,
    inputs: Optional[torch.Tensor] = None,
) -> PSQDecomp:
    """
    Compute a decomposition of a layer weight matrix using SVD.

    Args:
        layer_weight (torch.Tensor): The weight matrix of the layer.
        device (str | torch.device): Device on which to perform computations.
        inputs (Optional[torch.Tensor]): Optional input data for data-dependent SVD.

    Returns:
        `PSQDecomp`: The decomposition result.
    """
    layer_weight = layer_weight.to(device, torch.float32)
    if inputs is not None:
        inputs = inputs.to(device=device, dtype=torch.float32)
        if inputs.dim() > 2:
            inputs = inputs.reshape(-1, inputs.shape[-1])
        try:
            return _compute_datasvd(layer_weight, inputs, device)
        except LinAlgError:
            log.warning("DataSVD did not converge, using standard SVD.")

    # pylint: disable=not-callable
    U, S, Vh = torch.linalg.svd(layer_weight, full_matrices=False)
    return PSQDecomp(U, S, Vh)


@torch.no_grad()
def decompose_linear(
    layer: torch.nn.Linear,
    inputs: Optional[torch.Tensor] = None,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float32,
    skip_decomposition: bool = False,
) -> ODLinear:
    """
    Decompose a dense linear layer into an ``ODLinear`` factorization.

    There are two distinct modes here:

    1. ``skip_decomposition=True``: build an architecture-compatible
       ``ODLinear`` without inspecting the source weights.
    2. ``skip_decomposition=False``: compute the factorization first, then
       materialize the resulting UV factors directly into a fresh ``ODLinear``.

    In the real decomposition path, this function does not rely on
    ``build_decomp_linear_like()`` for anything other than the explicit skip
    branch. The actual factorization is computed from the source layer's dense
    weight matrix, optionally using representative inputs to drive the
    data-aware SVD variant.

    Args:
        layer: Dense linear layer to decompose.
        inputs: Optional sample inputs used for DataSVD.
        device: Target device for the decomposition work and the materialized
            decomposed layer. Defaults to the source layer device.
        dtype: Target dtype for the stored low-rank factors.
        skip_decomposition: If ``True``, return a randomly initialized
            ``ODLinear`` with matching shape instead of factorizing ``layer``.

    Returns:
        ``ODLinear`` whose factors reconstruct the source layer weight and whose
        bias, when present, is copied directly from the source layer.
    """
    if skip_decomposition:
        return build_decomp_linear_like(layer, device=device, dtype=dtype)

    weight = layer.weight.data
    target_device = device or layer.weight.device
    decomp = compute_decomp(weight, target_device, inputs)
    u_mat, v_mat = decomp.to_UV_decomp(dtype)

    decomp_layer = ODLinear.build_from_uv(
        u=v_mat.T,
        v=u_mat,
        bias=None if layer.bias is None else layer.bias.data.to(dtype),
        device=device,
        dtype=dtype,
    )
    decomp_layer.eigs = decomp.S
    return decomp_layer


@torch.no_grad()
def decompose_conv2d(
    layer: torch.nn.Conv2d,
    inputs: Optional[torch.Tensor] = None,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float32,
    skip_decomposition: bool = False,
) -> ODConv2d:
    """
    Decompose a dense ``Conv2d`` into a two-factor ``ODConv2d``.

    As with ``decompose_linear()``, this function separates the architecture-only
    path from the real decomposition path. When decomposition is enabled, the
    convolution kernel is first flattened into a matrix of shape
    ``(out_channels, in_channels * kernel_area)``, factorized, then reshaped
    back into the ``weight_u`` / ``weight_v`` kernels expected by ``ODConv2d``.

    Args:
        layer: Dense convolution layer to decompose.
        inputs: Optional sample inputs used for DataSVD.
        device: Target device for the decomposition work and the materialized
            decomposed layer. Defaults to the source layer device.
        dtype: Target dtype for the stored low-rank factors.
        skip_decomposition: If ``True``, return a randomly initialized
            ``ODConv2d`` with matching geometry instead of factorizing ``layer``.

    Returns:
        ``ODConv2d`` whose two kernels reconstruct the source convolution and
        whose bias, when present, is copied directly from the source layer.
    """
    if skip_decomposition:
        return build_decomp_conv2d_like(layer, device=device, dtype=dtype)

    weight = layer.weight.reshape(layer.out_channels, -1)
    target_device = device or layer.weight.device
    decomp = compute_decomp(weight, target_device, inputs)
    u_mat, v_mat = decomp.to_UV_decomp(dtype)

    decomp_layer = ODConv2d.build_from_uv(
        u=v_mat.T.reshape(-1, layer.in_channels, *layer.kernel_size),
        v=u_mat.reshape(layer.out_channels, -1, 1, 1),
        bias=None if layer.bias is None else layer.bias.data.to(dtype),
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        device=device,
        dtype=dtype,
    )
    decomp_layer.eigs = decomp.S
    return decomp_layer


def decompose(
    layer: nn.Linear | nn.Conv2d,
    name: str,
    inputs: Optional[torch.Tensor],
    distr: DistributedInfo,
    skip: bool = False,
) -> nn.Module:
    """Decompose a supported layer and move it back to the expected device."""
    layer_decomp_fns = {
        nn.Conv2d: decompose_conv2d,
        nn.Linear: decompose_linear,
    }

    svd_device = distr.device
    end_device = "cpu" if distr.is_fsdp else svd_device
    layer_dtype = layer.weight.dtype

    fn = layer_decomp_fns.get(type(layer))
    if fn is None:
        log.warning(
            "Unhandled type %s for %s: leaving layer non decomposed",
            type(layer),
            name,
        )
        return layer.to(end_device)

    return fn(
        layer,
        inputs,
        svd_device,
        dtype=layer_dtype,
        skip_decomposition=skip,
    ).to(end_device)


def build_decomp_like(
    layer: nn.Module,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
    **kwargs,
):
    """
    Build an empty decomposed counterpart for a supported dense layer type.

    This helper mirrors only architecture: feature sizes, convolution geometry,
    bias flag, and optional construction kwargs. It does not inspect or copy
    the source weights, and it is intentionally separate from the actual
    decomposition path used by ``decompose_linear()`` and ``decompose_conv2d()``.

    Args:
        layer: Source dense layer whose architecture should be mirrored.
        device: Device for the constructed decomposed layer. Defaults to the
            source layer device.
        dtype: Dtype for the constructed decomposed layer. Defaults to the
            source layer dtype.
        **kwargs: Additional constructor arguments forwarded to the layer-specific builder.

    Returns:
        An ``ODLinear`` or ``ODConv2d`` instance with matching architecture.
    """
    layer_builder_fns = {
        nn.Conv2d: build_decomp_conv2d_like,
        nn.Linear: build_decomp_linear_like,
    }

    end_device = device or layer.weight.device
    layer_dtype = dtype or layer.weight.dtype
    fn = layer_builder_fns.get(type(layer))

    assert fn is not None, f"Unhandled type {type(layer)}"
    return fn(layer, device=end_device, dtype=layer_dtype, **kwargs)
