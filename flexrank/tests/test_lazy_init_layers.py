"""Tests for lazy materialization of decomposed layers from UV factors."""

# pylint: disable=not-callable,duplicate-code

import pytest
import torch
import torch.nn.functional as F

from flexrank.layers.conv import ODConv2d
from flexrank.layers.decomposition import decompose_linear
from flexrank.layers.linear import ODLinear, build_decomp_linear_like


def test_odlinear_lazy_init_materializes_from_uv():
    """A lazy linear layer should stay empty until UV factors are materialized."""
    layer = ODLinear(5, 3, bias=True, lazy_init=True)

    assert not layer.is_initialized
    assert not list(layer.parameters())
    with pytest.raises(RuntimeError, match="from_uv"):
        layer(torch.randn(2, 5))

    u = torch.randn(2, 5)
    v = torch.randn(3, 2)
    bias = torch.randn(3)

    layer.from_uv(u=u, v=v, bias=bias)

    inputs = torch.randn(4, 5)
    expected = F.linear(F.linear(inputs, u), v, bias)

    assert layer.is_initialized
    assert layer.inner_dim == 2
    assert layer.max_inner_dim == 2
    assert torch.allclose(layer.weight, v @ u)
    assert torch.allclose(layer(inputs), expected)


def test_odlinear_build_from_uv_factory_and_lazy_builder():
    """Factory helpers should build a usable linear layer directly from UV factors."""
    dense = torch.nn.Linear(5, 3)
    lazy_layer = build_decomp_linear_like(dense, lazy_init=True)

    assert not lazy_layer.is_initialized
    assert not list(lazy_layer.parameters())

    u = torch.randn(2, 5)
    v = torch.randn(3, 2)
    bias = torch.randn(3)
    layer = ODLinear.build_from_uv(u=u, v=v, bias=bias)

    inputs = torch.randn(4, 5)
    expected = F.linear(F.linear(inputs, u), v, bias)

    assert layer.is_initialized
    assert torch.allclose(layer(inputs), expected)


def test_odlinear_from_uv_enforces_bias_contract():
    """Linear UV materialization should respect the bias choice fixed at construction."""
    u = torch.randn(2, 5)
    v = torch.randn(3, 2)
    bias = torch.randn(3)

    with pytest.raises(ValueError, match="constructed with bias=True"):
        ODLinear(5, 3, bias=True, lazy_init=True).from_uv(u=u, v=v, bias=None)

    with pytest.raises(ValueError, match="constructed with bias=False"):
        ODLinear(5, 3, bias=False, lazy_init=True).from_uv(u=u, v=v, bias=bias)


def test_odconv2d_lazy_init_materializes_from_uv():
    """A lazy convolution should stay empty until UV kernels are materialized."""
    layer = ODConv2d(
        3,
        5,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=True,
        lazy_init=True,
    )

    assert not layer.is_initialized
    assert not list(layer.parameters())
    with pytest.raises(RuntimeError, match="from_uv"):
        layer(torch.randn(2, 3, 8, 8))

    u = torch.randn(4, 3, 3, 3)
    v = torch.randn(5, 4, 1, 1)
    bias = torch.randn(5)
    layer.from_uv(u=u, v=v, bias=bias)

    inputs = torch.randn(2, 3, 8, 8)
    expected = F.conv2d(F.conv2d(inputs, u, None, stride=2, padding=1), v, bias)

    assert layer.is_initialized
    assert layer.inner_dim == 4
    assert layer.max_inner_dim == 4
    assert torch.allclose(layer(inputs), expected)


def test_odconv2d_from_uv_enforces_bias_contract():
    """Convolution UV materialization should respect the bias contract."""
    u = torch.randn(4, 3, 3, 3)
    v = torch.randn(5, 4, 1, 1)
    bias = torch.randn(5)

    with pytest.raises(ValueError, match="constructed with bias=True"):
        ODConv2d(3, 5, kernel_size=3, bias=True, lazy_init=True).from_uv(
            u=u,
            v=v,
            bias=None,
        )

    with pytest.raises(ValueError, match="constructed with bias=False"):
        ODConv2d(3, 5, kernel_size=3, bias=False, lazy_init=True).from_uv(
            u=u,
            v=v,
            bias=bias,
        )


def test_odconv2d_build_from_uv_factory():
    """The convolution UV factory should build a ready-to-run decomposed layer."""
    u = torch.randn(4, 3, 3, 3)
    v = torch.randn(5, 4, 1, 1)
    bias = torch.randn(5)
    layer = ODConv2d.build_from_uv(
        u=u,
        v=v,
        bias=bias,
        stride=2,
        padding=1,
    )

    inputs = torch.randn(2, 3, 8, 8)
    expected = F.conv2d(F.conv2d(inputs, u, None, stride=2, padding=1), v, bias)

    assert layer.is_initialized
    assert torch.allclose(layer(inputs), expected)


def test_decompose_linear_accepts_datasvd_covariance_inputs():
    """The covariance-based decomposition path should remain usable after refactors."""
    torch.manual_seed(0)
    dense = torch.nn.Linear(5, 3, bias=True)
    sample_inputs = torch.randn(16, 5)
    inputs_covar = sample_inputs.T @ sample_inputs

    layer = decompose_linear(dense, inputs=inputs_covar)

    assert layer.is_initialized
    assert layer.weight.shape == dense.weight.shape
    assert layer.eigs is not None
