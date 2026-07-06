"""Base abstractions for FlexRank decomposed layers."""

from enum import StrEnum, auto
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class ODImpl(StrEnum):
    """Forward implementations supported by decomposed layers."""

    DENSE = auto()
    SLICED = auto()
    GAR = auto()

    @property
    def code(self) -> int:
        """Tensor-serializable code derived from enum declaration order."""
        return tuple(type(self)).index(self)

    @classmethod
    def from_code(cls, code: int) -> "ODImpl":
        """Restore an implementation enum from its tensor-serializable code."""
        return tuple(cls)[code]


class ODLayer(Module):
    """Base class shared by FlexRank decomposed layers."""

    __constants__ = ["_in_features", "_out_features"]
    _in_features: int
    _out_features: int
    _max_inner_dim: int
    _inner_dim: int
    _frwd_impl: Callable[..., torch.Tensor]
    eigs: Optional[torch.Tensor]
    _impl: ODImpl | None
    supports_gar: bool = False

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features

        self._max_inner_dim = min(in_features, out_features)
        self._inner_dim = self._max_inner_dim
        self.eigs = None
        self._impl = None
        self._initialized = True

    @property
    def in_features(self):
        """Input feature dimension of the represented dense layer."""
        return self._in_features

    @property
    def out_features(self):
        """Output feature dimension of the represented dense layer."""
        return self._out_features

    @property
    def max_inner_dim(self):
        """Maximum rank or inner dimension supported by this layer."""
        return self._max_inner_dim

    @property
    def inner_dim(self):
        """Current active rank used by the layer."""
        return self._inner_dim

    @inner_dim.setter
    def inner_dim(self, inner_dim: int):
        """Set the active rank used by the layer."""
        inner_dim = int(inner_dim)
        assert inner_dim <= self._max_inner_dim, (
            f"Inner dimension cannot be greater than max_inner_dim {self._max_inner_dim}"
        )
        if self._impl is ODImpl.GAR and inner_dim != self._inner_dim:
            raise RuntimeError(
                "Cannot modify `inner_dim` after an ODLayer has been converted to GAR mode."
            )
        self._inner_dim = inner_dim
        self._on_inner_dim_changed()

    def _on_inner_dim_changed(self) -> None:
        """Hook for subclasses that keep derived runtime state."""

    @property
    def width(self):
        """Alias for the currently active inner dimension."""
        return self._inner_dim

    @property
    def impl(self) -> ODImpl | None:
        """Current forward/storage implementation used by the layer."""
        return self._impl

    def export_flexrank_state(self) -> dict:
        """Return JSON-serializable metadata needed to reconstruct this OD layer."""
        return {
            "class": type(self).__name__,
            "inner_dim": self.inner_dim,
            "max_inner_dim": self.max_inner_dim,
            "impl": self.impl.value,
        }

    def get_extra_state(self) -> Tensor:
        """Return tensor-only extra state for PyTorch/HF state_dict compatibility."""
        return torch.tensor(self._extra_state_tuple(), dtype=torch.int64)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Allow FSDP/HF reload paths to omit redundant OD extra state."""
        extra_state_key = prefix + "_extra_state"
        if extra_state_key not in state_dict:
            state_dict[extra_state_key] = {}
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _extra_state_tuple(self) -> tuple[int, int, int]:
        """Return the compact tensor state as primitive integers."""
        return self.inner_dim, self.max_inner_dim, self.impl.code

    @staticmethod
    def _parse_extra_state(state: dict | Tensor) -> tuple[int, int, ODImpl]:
        """Parse PyTorch tensor state or JSON config state."""
        if isinstance(state, Tensor):
            inner_dim, max_inner_dim, impl_id = [int(v) for v in state.tolist()]
            impl = ODImpl.from_code(impl_id)
        else:
            inner_dim = int(state["inner_dim"])
            max_inner_dim = int(state["max_inner_dim"])
            impl = ODImpl(state["impl"])

        if inner_dim > max_inner_dim:
            raise ValueError(f"inner_dim={inner_dim} cannot exceed max_inner_dim={max_inner_dim}")
        return inner_dim, max_inner_dim, impl

    def _set_impl(self, impl: ODImpl) -> None:
        """Set the implementation marker. Subclasses also update forward functions."""
        self._impl = impl

    def set_extra_state(self, state: dict | Tensor) -> None:
        """Restore serializable OD metadata."""
        if state is None or (isinstance(state, dict) and not state):
            return

        inner_dim, max_inner_dim, impl = self._parse_extra_state(state)
        self._max_inner_dim = max_inner_dim
        self._inner_dim = inner_dim
        self._set_impl(impl)

    def materialize_runtime_buffers(self) -> None:
        """Materialize derived non-persistent buffers after checkpoint loading."""

    def get_num_parameters(self, inner_dim: Optional[int] = None) -> int:
        """Return the parameter count for a given inner dimension."""
        raise NotImplementedError

    @property
    def num_parameters(self) -> int:
        """Return the parameter count for the current inner dimension."""
        return self.get_num_parameters()

    def get_weight(self, inner_dim: Optional[int] = None):
        """Materialize the equivalent dense weight for a given inner dimension."""
        raise NotImplementedError

    def reset_parameters(self) -> None:
        """Reinitialize the stored factor parameters."""
        raise NotImplementedError

    @property
    def weight(self):
        """Materialize the equivalent dense weight at the current rank."""
        return self.get_weight()

    @property
    def is_initialized(self) -> bool:
        """Whether the layer factors have been materialized."""
        return self._initialized

    def _require_initialized(self) -> None:
        """Raise when an operation needs materialized factor tensors."""
        if self._initialized:
            return

        raise RuntimeError(
            f"{type(self).__name__} has no materialized parameters. "
            "Call from_uv(...) first or construct it with lazy_init=False."
        )

    @staticmethod
    def _as_parameter(
        tensor: Tensor,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Parameter:
        """Wrap a tensor as a parameter, optionally moving/casting it first."""
        if device is not None or dtype is not None:
            tensor = tensor.to(
                device=device or tensor.device,
                dtype=dtype or tensor.dtype,
            )
        return Parameter(tensor.detach())

    def prune_weights(self, use_gar: bool = False) -> None:
        """Permanently shrink parameter tensors to the active inner dimension."""
        raise NotImplementedError
