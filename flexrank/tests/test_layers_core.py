"""Coverage for public decomposed layer behavior."""

# pylint: disable=missing-function-docstring

import pytest
import torch

from flexrank.layers.base import ODImpl
from flexrank.layers.conv import ODConv2d
from flexrank.layers.decomposition import (
    build_decomp_like,
    decompose,
    decompose_conv2d,
    decompose_linear,
)
from flexrank.layers.linear import ODLinear
from flexrank.types import get_single_worker_distributed_info


def test_odimpl_code_roundtrip_and_extra_state_validation():
    assert [ODImpl.from_code(impl.code) for impl in ODImpl] == list(ODImpl)

    layer = ODLinear(3, 2)
    with pytest.raises(ValueError, match="inner_dim=3"):
        layer.set_extra_state({"inner_dim": 3, "max_inner_dim": 2, "impl": ODImpl.DENSE.value})


def test_odlinear_forward_matches_base_linear_for_dense_and_sliced_impls():
    u = torch.tensor(
        [
            [1.0, 2.0, -1.0, 0.5],
            [0.0, -1.0, 3.0, 2.0],
            [1.5, 0.25, -0.5, 1.0],
        ]
    )
    v = torch.tensor(
        [
            [2.0, -1.0, 0.5],
            [0.5, 1.0, -0.25],
            [-2.0, 0.25, 1.5],
        ]
    )
    bias = torch.tensor([0.1, -0.2, 0.3])
    inputs = torch.randn(5, 4)
    active_rank = 2

    dense = ODLinear.build_from_uv(u=u, v=v, bias=bias, frwd_impl=ODImpl.DENSE)
    sliced = ODLinear.build_from_uv(u=u, v=v, bias=bias, frwd_impl=ODImpl.SLICED)
    dense.inner_dim = active_rank
    sliced.inner_dim = active_rank

    base = torch.nn.Linear(4, 3)
    assert base.bias is not None
    with torch.no_grad():
        base.weight.copy_(v[:, :active_rank] @ u[:active_rank, :])
        base.bias.copy_(bias)
    expected = base(inputs)

    assert torch.allclose(dense(inputs), expected)
    assert torch.allclose(sliced(inputs), expected)
    assert torch.allclose(dense(inputs), sliced(inputs))
    assert dense.get_num_parameters() == 2 * 4 + 3 * 2 - 2 * 2 + 3
    assert dense.export_flexrank_state() == {
        "class": "ODLinear",
        "inner_dim": active_rank,
        "max_inner_dim": 3,
        "impl": ODImpl.DENSE.value,
    }


def test_odlinear_prune_and_torch_conversion_preserve_outputs():
    u = torch.tensor([[2.0, 0.0, 1.0], [0.5, -1.0, 1.5], [3.0, 1.0, -0.5]])
    v = torch.tensor([[1.0, -2.0, 0.5], [0.0, 1.5, -1.0]])
    bias = torch.tensor([0.25, -0.5])
    layer = ODLinear.build_from_uv(u=u, v=v, bias=bias)
    layer.inner_dim = 2
    inputs = torch.randn(4, 3)
    expected = layer(inputs).clone()

    torch_layer = layer.to_torch_linear()
    layer.prune_weights()

    assert layer.impl is ODImpl.DENSE
    assert layer.max_inner_dim == 2
    assert layer.weight_u.shape == (2, 3)
    assert layer.weight_v.shape == (2, 2)
    assert torch.allclose(layer(inputs), expected)
    assert torch.allclose(torch_layer(inputs), expected)


def test_odconv2d_forward_matches_base_conv2d_for_dense_and_sliced_impls():
    u = torch.randn(3, 2, 3, 3)
    v = torch.randn(3, 3, 1, 1)
    bias = torch.randn(3)
    inputs = torch.randn(4, 2, 6, 6)
    active_rank = 2

    dense = ODConv2d.build_from_uv(u=u, v=v, bias=bias, padding=1)
    sliced = ODConv2d.build_from_uv(
        u=u,
        v=v,
        bias=bias,
        padding=1,
        frwd_impl=ODImpl.SLICED,
    )
    dense.inner_dim = active_rank
    sliced.inner_dim = active_rank

    base = torch.nn.Conv2d(2, 3, kernel_size=3, padding=1)
    assert base.bias is not None
    with torch.no_grad():
        base.weight.copy_(
            (
                v[:, :active_rank, :, :].reshape(3, active_rank)
                @ u[:active_rank, :, :, :].reshape(active_rank, -1)
            ).reshape(3, 2, 3, 3)
        )
        base.bias.copy_(bias)
    expected = base(inputs)

    assert torch.allclose(dense(inputs), expected, atol=1e-6)
    assert torch.allclose(sliced(inputs), expected, atol=1e-6)
    assert torch.allclose(dense(inputs), sliced(inputs), atol=1e-6)
    assert dense.get_num_parameters() == 2 * 2 * 3 * 3 + 3 * 2 - 2 * 2 + 3


def test_odconv2d_validation_and_pruning_preserve_outputs():
    with pytest.raises(NotImplementedError):
        ODConv2d(2, 4, kernel_size=3, groups=2)

    layer = ODConv2d.build_from_uv(
        u=torch.randn(3, 2, 3, 3),
        v=torch.randn(4, 3, 1, 1),
        bias=torch.randn(4),
        padding=1,
    )
    layer.inner_dim = 2
    inputs = torch.randn(2, 2, 5, 5)
    expected = layer(inputs).clone()
    weight = layer.get_weight().clone()

    layer.prune_weights()

    assert layer.max_inner_dim == 2
    assert layer.weight_u.shape == (2, 2, 3, 3)
    assert layer.weight_v.shape == (4, 2, 1, 1)
    assert torch.allclose(layer.get_weight(), weight)
    assert torch.allclose(layer(inputs), expected)


def test_decomposition_helpers_reconstruct_supported_layers():
    linear = torch.nn.Linear(4, 3, bias=True)
    inputs = torch.randn(6, 4)
    decomp_linear = decompose_linear(linear)

    assert isinstance(build_decomp_like(linear, lazy_init=True), ODLinear)
    assert torch.allclose(decomp_linear.weight, linear.weight, atol=1e-5)
    assert torch.allclose(decomp_linear(inputs), linear(inputs), atol=1e-5)
    assert torch.allclose(decomp_linear.bias, linear.bias)
    assert decomp_linear.eigs is not None

    conv = torch.nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    conv_inputs = torch.randn(2, 2, 5, 5)
    decomp_conv = decompose_conv2d(conv)

    assert isinstance(build_decomp_like(conv, lazy_init=True), ODConv2d)
    assert torch.allclose(decomp_conv.weight, conv.weight, atol=1e-5)
    assert torch.allclose(decomp_conv(conv_inputs), conv(conv_inputs), atol=1e-5)
    assert torch.allclose(decomp_conv.bias, conv.bias)
    assert decomp_conv.eigs is not None


def test_decompose_dispatch_leaves_unsupported_modules_unchanged():
    module = torch.nn.BatchNorm1d(3)
    distr = get_single_worker_distributed_info()

    assert decompose(module, "norm", None, distr) is module
