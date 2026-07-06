"""Decomposed convolution layer implementation."""

# pylint: disable=duplicate-code,too-many-arguments,too-many-instance-attributes
# pylint: disable=too-many-positional-arguments

import math
from math import prod
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter, init

from flexrank.layers.base import ODImpl, ODLayer
from flexrank.layers.functional import (
    conv2d_slice_in,
    conv2d_slice_out,
    od_dense_conv2d,
    od_sliced_conv2d,
)

_FORWARD_IMPL = {ODImpl.DENSE: od_dense_conv2d, ODImpl.SLICED: od_sliced_conv2d}


def _as_pair(value: tuple[int, ...] | int) -> tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


class ODConv2d(ODLayer):
    """Two-stage convolution with adjustable active rank."""

    __constants__ = [
        "_in_features",
        "_out_features",
        "_stride",
        "_padding",
        "_dilation",
        "_groups",
        "_kernel_size",
    ]

    _kernel_size: tuple[int, ...] | int
    _stride: tuple[int, ...] | int
    _padding: str | tuple[int, ...] | int
    _dilation: tuple[int, ...] | int
    _groups: int

    weight_u: Parameter | None
    weight_v: Parameter | None
    bias: Parameter | None
    _rank_mask: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        dilation: int | tuple = 1,
        groups: int = 1,
        bias: bool = True,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        frwd_impl: ODImpl = ODImpl.DENSE,
        lazy_init: bool = False,
    ):

        super().__init__(in_features, out_features)
        kernel_total_size = prod(kernel_size) if isinstance(kernel_size, tuple) else kernel_size**2
        self._max_inner_dim = min(out_features, in_features * kernel_total_size)
        self._inner_dim = self._max_inner_dim

        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._kernel_size = kernel_size
        self._groups = groups
        self._use_bias = bias
        self._factory_device = device
        self._factory_dtype = dtype
        self.register_parameter("weight_u", None)
        self.register_parameter("weight_v", None)
        self.register_parameter("bias", None)

        if groups != 1:
            raise NotImplementedError

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

    def _resolved_factory_kwargs(self) -> dict[str, Optional[torch.device | torch.dtype]]:
        """Resolve the target device and dtype for parameter materialization."""
        device = self._factory_device
        dtype = self._factory_dtype
        weight_u = getattr(self, "weight_u", None)
        if device is None and weight_u is not None:
            device = weight_u.device
        if dtype is None and weight_u is not None:
            dtype = weight_u.dtype
        return {"device": device, "dtype": dtype}

    def _init_factors(self, *, inner_dim: int, bias: bool) -> None:
        """Instantiate factor parameters and initialize them like `nn.Conv2d`."""
        factory_kwargs = self._resolved_factory_kwargs()
        self.weight_u = self._as_parameter(
            torch.empty(
                inner_dim,
                self._in_features,
                *_as_pair(self._kernel_size),
                **factory_kwargs,
            )
        )
        self.weight_v = self._as_parameter(
            torch.empty(self._out_features, inner_dim, 1, 1, **factory_kwargs)
        )
        if bias:
            self.bias = self._as_parameter(torch.empty(self._out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self._max_inner_dim = inner_dim
        self._inner_dim = inner_dim
        self._use_bias = bias
        self._reset_rank_mask()
        self._initialized = True
        self.reset_parameters()

    @classmethod
    def build_from_uv(
        cls,
        u: Tensor,
        v: Tensor,
        bias: Optional[Tensor] = None,
        *,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        dilation: int | tuple = 1,
        groups: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        frwd_impl: ODImpl = ODImpl.DENSE,
    ) -> "ODConv2d":
        """Build and materialize a decomposed convolution from factor tensors."""
        if u.ndim != 4 or v.ndim != 4:
            raise ValueError("Expected `u` and `v` to be rank-4 convolution kernels.")

        layer = cls(
            in_features=u.shape[1],
            out_features=v.shape[0],
            kernel_size=u.shape[2:],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias is not None,
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
    ) -> "ODConv2d":
        """Materialize factor tensors from precomputed UV convolution kernels."""
        if u.ndim != 4 or v.ndim != 4:
            raise ValueError("Expected `u` and `v` to be rank-4 convolution kernels.")

        inner_dim, in_features, *kernel_size = u.shape
        out_features, v_inner_dim, v_kernel_height, v_kernel_width = v.shape
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
        if tuple(kernel_size) != _as_pair(self.kernel_size):
            raise ValueError(f"Expected kernel size {self.kernel_size}, got {tuple(kernel_size)}.")
        if (v_kernel_height, v_kernel_width) != (1, 1):
            raise ValueError(
                f"Expected `v` to be a 1x1 kernel, got {(v_kernel_height, v_kernel_width)}."
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

    def _resize_empty_storage(self, inner_dim: int) -> None:
        """Prepare parameter shapes before checkpoint tensors are loaded."""
        factory_kwargs = self._resolved_factory_kwargs()
        self.weight_u = self._as_parameter(
            torch.empty(
                inner_dim,
                self.in_features,
                *_as_pair(self.kernel_size),
                **factory_kwargs,
            )
        )
        self.weight_v = self._as_parameter(
            torch.empty(self.out_features, inner_dim, 1, 1, **factory_kwargs)
        )

    @property
    def kernel_size(self) -> tuple[int, ...] | int:
        """Kernel size of the input-side factor convolution."""
        return self._kernel_size

    @property
    def stride(self) -> tuple[int, ...] | int:
        """Stride used by the input-side factor convolution."""
        return self._stride

    @property
    def padding(self) -> str | tuple[int, ...] | int:
        """Padding used by the input-side factor convolution."""
        return self._padding

    @property
    def dilation(self) -> tuple[int, ...] | int:
        """Dilation used by the input-side factor convolution."""
        return self._dilation

    @property
    def groups(self) -> int:
        """Grouping configuration for the factor convolution."""
        return self._groups

    def _slice_u(self, inner_dim: Optional[int] = None) -> Tensor:
        self._require_initialized()
        inner_dim = inner_dim or self._inner_dim
        return conv2d_slice_in(self.weight_u, inner_dim)

    def _slice_v(self, inner_dim: Optional[int] = None) -> Tensor:
        self._require_initialized()
        inner_dim = inner_dim or self._inner_dim
        return conv2d_slice_out(self.weight_v, inner_dim)

    def get_weight(self, inner_dim: Optional[int] = None):
        """Materialize the equivalent dense convolution kernel."""
        self._require_initialized()
        inner_dim = inner_dim or self._inner_dim
        weight = self._slice_v(inner_dim).reshape(self._out_features, -1) @ self._slice_u(
            inner_dim
        ).reshape(inner_dim, -1)
        return weight.reshape(
            self._out_features,
            self._in_features,
            *_as_pair(self._kernel_size),
        )

    def get_num_parameters(self, inner_dim: Optional[int] = None) -> int:
        self._require_initialized()
        inner_dim = inner_dim or self._inner_dim
        uk_h, uk_w = _as_pair(self._kernel_size)
        n_params = (
            inner_dim * self._in_features * uk_h * uk_w
            + self._out_features * inner_dim
            - inner_dim * inner_dim
        )
        if self.bias is not None:
            n_params += self.bias.numel()
        return n_params

    def reset_parameters(self) -> None:
        """Reset both factor convolutions using `nn.Conv2d`-style initialization."""
        self._require_initialized()
        init.kaiming_uniform_(self.weight_u, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight_v.shape[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def prune_weights(self, use_gar: bool = False):
        self._require_initialized()
        if use_gar:
            raise NotImplementedError("GAR mode is not implemented for conv layers")
        self.weight_u = Parameter(self._slice_u().data)
        self.weight_v = Parameter(self._slice_v().data)
        self._max_inner_dim = self._inner_dim
        self._reset_rank_mask()

    def set_extra_state(self, state: dict | Tensor) -> None:
        """Restore ODConv2d metadata and prepare checkpoint tensor shapes."""
        if state is None or (isinstance(state, dict) and not state):
            return

        inner_dim, max_inner_dim, impl = self._parse_extra_state(state)
        if impl is ODImpl.GAR:
            raise NotImplementedError("GAR mode is not implemented for conv layers")

        if max_inner_dim != self.max_inner_dim:
            self._resize_empty_storage(max_inner_dim)

        self._max_inner_dim = max_inner_dim
        self._inner_dim = inner_dim
        self._set_impl(impl)
        self._reset_rank_mask()
        self._initialized = True

    def extra_repr(self) -> str:
        """Return a compact string representation for debugging."""
        return (
            f"in_features={self._in_features}, "
            f"out_features={self._out_features}, "
            f"kernel_size={self._kernel_size}, "
            f"stride={self._stride}, "
            f"padding={self._padding}, "
            f"bias={self._use_bias}, "
            f"current_rank={self._inner_dim}, "
            f"max_rank={self._max_inner_dim}, "
            f"initialized={self.is_initialized}"
        )

    def _forward_zeros(self, shape: torch.Size) -> Tensor:
        """Return the bias-only output used when the active rank is zero."""
        self._require_initialized()
        batch_size, _, height, width = shape
        stride = _as_pair(self._stride)
        padding = _as_pair(self._padding)
        dilation = _as_pair(self._dilation)
        kernel_height, kernel_width = _as_pair(self._kernel_size)

        out_height = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[
            0
        ] + 1
        out_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1

        out = torch.zeros(
            (batch_size, self._out_features, out_height, out_width),
            dtype=self.weight_v.dtype,
            device=self.weight_v.device,
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the selected low-rank convolution implementation."""
        self._require_initialized()
        return self._frwd_impl(
            inputs,
            self.weight_u,
            self.weight_v,
            self._inner_dim,
            self.bias,
            self._stride,
            self._padding,
            self._dilation,
            self._rank_mask,
        )


def build_decomp_conv2d_like(
    layer: torch.nn.Conv2d,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> ODConv2d:
    """Create an `ODConv2d` with the same architecture as an existing `nn.Conv2d`."""
    device = device or layer.weight.device

    return ODConv2d(
        in_features=layer.in_channels,
        out_features=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=layer.bias is not None,
        device=device,
        dtype=dtype,
        frwd_impl=kwargs.pop("frwd_impl", ODImpl.DENSE),
        lazy_init=kwargs.pop("lazy_init", False),
    )
