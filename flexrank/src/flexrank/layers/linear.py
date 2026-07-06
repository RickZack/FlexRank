"""Decomposed linear layer implementation."""

# pylint: disable=duplicate-code,too-many-arguments,too-many-instance-attributes
# pylint: disable=too-many-positional-arguments

import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter, init

from flexrank.layers.base import ODImpl, ODLayer
from flexrank.layers.functional import (
    linear_slice_in,
    linear_slice_out,
    od_dense_linear,
    od_sliced_linear,
)
from flexrank.layers.gar import make_gar, od_gar_linear

_FORWARD_IMPL = {
    ODImpl.DENSE: od_dense_linear,
    ODImpl.SLICED: od_sliced_linear,
    ODImpl.GAR: od_gar_linear,
}


class ODLinear(ODLayer):
    """Two-factor linear layer with adjustable active rank."""

    weight_u: Parameter | None
    weight_v: Parameter | None
    bias: Parameter | None
    _rank_mask: Tensor
    supports_gar: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        frwd_impl: ODImpl = ODImpl.DENSE,
        lazy_init: bool = False,
    ):

        super().__init__(in_features, out_features)
        self._alpha = alpha
        self._use_bias = bias
        self._factory_device = device
        self._factory_dtype = dtype
        self.register_parameter("weight_u", None)
        self.register_parameter("weight_v", None)
        self.register_parameter("bias", None)

        if alpha is None:
            self.scaling = 1.0
        else:
            self.scaling = alpha / self.max_inner_dim

        if dropout > 0:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = torch.nn.Identity()

        if frwd_impl is ODImpl.GAR:
            raise ValueError(
                "Cannot use GAR forward because the weights are not in GAR form. "
                "Call layer.prune_weights(use_gar=True) to switch to GAR mode."
            )
        self._set_impl(frwd_impl)
        self.register_buffer(
            "_rank_mask",
            self._make_rank_mask(),
            persistent=False,
        )
        self._on_inner_dim_changed()

        if lazy_init:
            self._initialized = False
        else:
            self._init_factors(inner_dim=self.max_inner_dim, bias=bias)

    def _update_scaling(self) -> None:
        """Refresh the LoRA-like scaling factor after rank changes."""
        if self._alpha is None:
            self.scaling = 1.0
            return
        self.scaling = self._alpha / self.max_inner_dim

    def _resolved_factory_kwargs(self) -> dict[str, Optional[torch.device | torch.dtype]]:
        """Resolve the target device/dtype for parameter materialization."""
        device = self._factory_device
        dtype = self._factory_dtype
        weight_u = getattr(self, "weight_u", None)
        if device is None and weight_u is not None:
            device = weight_u.device
        if dtype is None and weight_u is not None:
            dtype = weight_u.dtype
        return {"device": device, "dtype": dtype}

    def _init_factors(self, *, inner_dim: int, bias: bool) -> None:
        """Instantiate factor parameters and initialize them like `nn.Linear`."""
        factory_kwargs = self._resolved_factory_kwargs()
        self.weight_u = self._as_parameter(
            torch.empty(inner_dim, self.in_features, **factory_kwargs)
        )
        self.weight_v = self._as_parameter(
            torch.empty(self.out_features, inner_dim, **factory_kwargs)
        )
        if bias:
            self.bias = self._as_parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self._max_inner_dim = inner_dim
        self._inner_dim = inner_dim
        self._use_bias = bias
        self._reset_rank_mask()
        self._update_scaling()
        self._initialized = True
        self.reset_parameters()

    @classmethod
    def build_from_uv(
        cls,
        u: Tensor,
        v: Tensor,
        bias: Optional[Tensor] = None,
        *,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        frwd_impl: ODImpl = ODImpl.DENSE,
    ) -> "ODLinear":
        """Build and materialize a decomposed linear layer from factor tensors."""
        if u.ndim != 2 or v.ndim != 2:
            raise ValueError("Expected `u` and `v` to be rank-2 tensors.")

        layer = cls(
            in_features=u.shape[1],
            out_features=v.shape[0],
            bias=bias is not None,
            alpha=alpha,
            dropout=dropout,
            device=device,
            dtype=dtype,
            frwd_impl=frwd_impl,
            lazy_init=True,
        )
        return layer.from_uv(u=u, v=v, bias=bias)

    def from_uv(
        self,
        *,
        u: Tensor,
        v: Tensor,
        bias: Optional[Tensor] = None,
    ) -> "ODLinear":
        """Materialize factor tensors from precomputed UV parameters."""
        if self.impl is ODImpl.GAR:
            raise RuntimeError("Cannot rematerialize UV factors after GAR conversion.")
        if u.ndim != 2 or v.ndim != 2:
            raise ValueError("Expected `u` and `v` to be rank-2 tensors.")

        inner_dim, in_features = u.shape
        out_features, v_inner_dim = v.shape
        if v_inner_dim != inner_dim:
            raise ValueError(
                f"Mismatched inner dimension between `u` ({inner_dim}) and `v` ({v_inner_dim})."
            )
        if in_features != self.in_features:
            raise ValueError(
                "`u` expects in_features="
                f"{in_features}, but layer was built for {self.in_features}."
            )
        if out_features != self.out_features:
            raise ValueError(
                "`v` expects out_features="
                f"{out_features}, but layer was built for {self.out_features}."
            )
        if self._use_bias and bias is None:
            raise ValueError(
                "This layer was constructed with bias=True, so from_uv(...) requires a bias tensor."
            )
        if not self._use_bias and bias is not None:
            raise ValueError(
                "This layer was constructed with bias=False, "
                "so from_uv(...) cannot accept a bias tensor."
            )
        if bias is not None and bias.shape != (self.out_features,):
            raise ValueError(
                f"Expected bias shape ({self.out_features},), got {tuple(bias.shape)}."
            )

        self._max_inner_dim = inner_dim
        self._inner_dim = inner_dim
        self._update_scaling()

        factory_kwargs = self._resolved_factory_kwargs()
        self.weight_u = self._as_parameter(u, **factory_kwargs)
        self.weight_v = self._as_parameter(v, **factory_kwargs)
        if bias is not None:
            self.bias = self._as_parameter(bias, **factory_kwargs)
        else:
            self.register_parameter("bias", None)

        self._reset_rank_mask()
        self._initialized = True
        return self

    def _set_impl(self, impl: ODImpl) -> None:
        """Set the storage/forward implementation."""
        self._impl = impl
        self._frwd_impl = _FORWARD_IMPL[impl]

    def _rank_mask_device_dtype(
        self,
    ) -> tuple[Optional[torch.device], Optional[torch.dtype]]:
        """Resolve device and dtype for rank-mask allocation."""
        weight_u = self.weight_u
        return (
            weight_u.device if weight_u is not None else self._factory_device,
            weight_u.dtype if weight_u is not None else self._factory_dtype,
        )

    def _make_rank_mask(self) -> Tensor:
        """Allocate a rank mask with the current maximum rank."""
        device, dtype = self._rank_mask_device_dtype()
        return torch.empty(
            self.max_inner_dim,
            device=device,
            dtype=dtype,
        )

    def _reset_rank_mask(self) -> None:
        """Recreate the static-shape dense mask after storage shape changes."""
        self._rank_mask = self._make_rank_mask()
        self._on_inner_dim_changed()

    def _on_inner_dim_changed(self) -> None:
        """Update the dense active-rank mask without changing its tensor metadata."""
        rank_mask = self._rank_mask
        with torch.no_grad():
            rank_mask.zero_()
            if self._inner_dim > 0:
                rank_mask[: self._inner_dim].fill_(1)

    def materialize_runtime_buffers(self) -> None:
        """Rebuild derived non-persistent buffers left on meta after HF loading."""
        if self._rank_mask.is_meta:
            self._reset_rank_mask()

    def _resize_empty_storage(self, impl: ODImpl, inner_dim: int) -> None:
        """Prepare parameter shapes before checkpoint tensors are loaded."""
        factory_kwargs = self._resolved_factory_kwargs()
        self.weight_u = self._as_parameter(
            torch.empty(inner_dim, self.in_features, **factory_kwargs)
        )
        if impl is ODImpl.GAR:
            self.weight_v = self._as_parameter(
                torch.empty(self.out_features - inner_dim, inner_dim, **factory_kwargs)
            )
        else:
            self.weight_v = self._as_parameter(
                torch.empty(self.out_features, inner_dim, **factory_kwargs)
            )

    def get_weight(self, inner_dim: Optional[int] = None):
        """Materialize the effective dense weight at the requested rank."""
        self._require_initialized()
        assert inner_dim is None or inner_dim <= self._max_inner_dim, (
            f"Inner dimension cannot be greater than max_inner_dim {self._max_inner_dim}"
        )
        inner_dim = inner_dim or self._inner_dim
        return self._slice_out(inner_dim) @ self._slice_in(inner_dim) * self.scaling

    def get_num_parameters(self, inner_dim: Optional[int] = None) -> int:
        self._require_initialized()
        inner_dim = inner_dim or self._inner_dim
        n_params = (
            inner_dim * self._in_features + self._out_features * inner_dim - inner_dim * inner_dim
        )
        if self.bias is not None:
            n_params += self.bias.numel()
        return n_params

    def reset_parameters(self) -> None:
        """Reset both factor matrices using `nn.Linear`-style initialization."""
        self._require_initialized()
        if self.impl is ODImpl.GAR:
            raise RuntimeError("Cannot reset parameters while using GAR storage.")
        init.kaiming_uniform_(self.weight_u, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight_v.shape[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _slice_in(self, inner_dim: Optional[int] = None) -> Tensor:
        """Return the active input-side factor."""
        self._require_initialized()
        if self.impl is ODImpl.GAR:
            return self.weight_u
        inner_dim = inner_dim or self._inner_dim
        return linear_slice_in(self.weight_u, inner_dim)

    def _slice_out(self, inner_dim: Optional[int] = None) -> Tensor:
        """Return the effective output-side factor at the active rank."""
        self._require_initialized()
        inner_dim = inner_dim or self._inner_dim
        if self.impl is ODImpl.GAR:
            return torch.cat(
                [
                    torch.eye(
                        inner_dim,
                        dtype=self.weight_v.dtype,
                        device=self.weight_v.device,
                    ),
                    self.weight_v,
                ],
                dim=0,
            )
        return linear_slice_out(self.weight_v, inner_dim)

    @torch.no_grad()
    def prune_weights(self, use_gar: bool = False):
        """Materialize the current rank-truncated weights into the stored parameters."""
        self._require_initialized()
        if self.impl is ODImpl.GAR:
            raise RuntimeError(
                "Cannot prune an ODLinear layer after it has been converted to GAR mode."
            )

        if use_gar:
            gar = make_gar(self._slice_out(), self._slice_in().T, self._inner_dim).gar
            self.weight_u = Parameter(gar.v.T.contiguous())
            self.weight_v = Parameter(gar.u.contiguous())
            self._impl = ODImpl.GAR
            self._frwd_impl = _FORWARD_IMPL[ODImpl.GAR]
        else:
            self.weight_u = Parameter(self._slice_in().data)
            self.weight_v = Parameter(self._slice_out().data)
            self._impl = ODImpl.SLICED if self.impl is not ODImpl.DENSE else ODImpl.DENSE
            self._frwd_impl = _FORWARD_IMPL[self.impl]

        self._max_inner_dim = self._inner_dim
        self._reset_rank_mask()
        self._update_scaling()

    def set_extra_state(self, state: dict | Tensor) -> None:
        """Restore ODLinear metadata and prepare checkpoint tensor shapes."""
        if state is None or (isinstance(state, dict) and not state):
            return

        inner_dim, max_inner_dim, impl = self._parse_extra_state(state)

        if impl is ODImpl.GAR or max_inner_dim != self.max_inner_dim:
            storage_dim = inner_dim if impl is ODImpl.GAR else max_inner_dim
            self._resize_empty_storage(impl, storage_dim)

        self._max_inner_dim = max_inner_dim
        self._inner_dim = inner_dim
        self._set_impl(impl)
        self._reset_rank_mask()
        self._update_scaling()
        self._initialized = True

    def extra_repr(self) -> str:
        return (
            f"in_features={self._in_features}, "
            f"out_features={self._out_features}, "
            f"bias={self._use_bias}, "
            f"current_rank={self._inner_dim}, "
            f"max_rank={self._max_inner_dim}, "
            f"initialized={self.is_initialized}, "
            f"impl={self.impl}"
        )

    @torch.no_grad()
    def to_torch_linear(self, device: Optional[str | torch.device] = None) -> torch.nn.Linear:
        """Convert this decomposed layer back into a standard `nn.Linear`."""
        self._require_initialized()
        device = device or self.weight_u.device
        full_layer = torch.nn.Linear(
            self._in_features,
            self._out_features,
            self.bias is not None,
        )
        full_layer.weight.data = self.weight.data
        if self.bias is not None:
            full_layer.bias.data = self.bias.data.clone()
        return full_layer.to(device)

    def _forward_zeros(self, shape: torch.Size) -> Tensor:
        """Return the bias-only output used when the active rank is zero."""
        self._require_initialized()
        out_shape = (*shape[:-1], self._out_features)
        out = torch.zeros(out_shape, dtype=self.weight_u.dtype, device=self.weight_u.device)
        if self.bias is not None:
            out += self.bias.view(*([1] * (out.dim() - 1)), -1)
        return out

    def _forward(self, inputs: Tensor) -> Tensor:
        """Run the selected low-rank forward implementation."""
        self._require_initialized()
        return self._frwd_impl(
            inputs,
            self.weight_u,
            self.weight_v,
            self._inner_dim,
            self.bias,
            self._rank_mask,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply dropout and then the selected low-rank linear transform."""
        inputs = self.dropout(inputs)
        return self._forward(inputs) * self.scaling


def build_decomp_linear_like(
    layer: torch.nn.Linear,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
):
    """
    Create an `ODLinear` with the same architecture as an existing `nn.Linear`.

    Args:
        layer (nn.Linear): The linear layer to replicate the configuration from.
        device (Optional[str | torch.device]): The device to place the new layer on.
            Defaults to the device of the input layer's weights.
        dtype (torch.dtype): The data type for the layer's parameters.
            Defaults to the data type of the input layer's weights.
        **kwargs: Additional keyword arguments to pass to ODLinear.

    Returns:
        ODLinear: A new decomposed linear layer with matching dimensions and bias.
    """
    device = device or layer.weight.device
    dtype = dtype or layer.weight.dtype

    return ODLinear(
        layer.in_features,
        layer.out_features,
        layer.bias is not None,
        alpha=kwargs.pop("alpha", None),
        dropout=kwargs.pop("dropout", 0.0),
        device=device,
        dtype=dtype,
        frwd_impl=kwargs.pop("frwd_impl", ODImpl.DENSE),
        lazy_init=kwargs.pop("lazy_init", False),
    )
