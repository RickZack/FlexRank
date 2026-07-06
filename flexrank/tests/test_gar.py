"""Unit tests for the GAR reparameterization helpers."""

# pylint: disable=not-callable

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import pytest
import torch
from flexrank.layers.decomposition import decompose_linear


def _load_gar_module():
    """Load `gar.py` directly to avoid package import side effects in tests."""
    gar_path = Path(__file__).resolve().parents[1] / "src" / "flexrank" / "layers" / "gar.py"
    spec = importlib.util.spec_from_file_location("test_gar_module", gar_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_GAR_MODULE = _load_gar_module()
make_gar = _GAR_MODULE.make_gar


def _load_module(name: str, relative_path: str):
    """Load a module from a flexrank package-relative path."""
    module_path = Path(__file__).resolve().parents[1] / "src" / "flexrank" / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_BASE_MODULE = _load_module("test_base_module", "layers/base.py")
_LINEAR_MODULE = _load_module("test_linear_module", "layers/linear.py")

ODImpl = _BASE_MODULE.ODImpl
ODLinear = _LINEAR_MODULE.ODLinear


def _reconstruct_left_factor(result, rank):
    """Reconstruct the full left factor from the compact GAR form."""
    return torch.cat(
        [
            torch.eye(rank, dtype=result.gar.u.dtype, device=result.gar.u.device),
            result.gar.u,
        ],
        dim=0,
    )


def _make_factor_with_invertible_leading_block(m, n, inner_dim, rank):
    """Construct factors whose leading ``r x r`` block is safely invertible."""
    generator = torch.Generator().manual_seed(0)
    u = torch.randn(m, inner_dim, generator=generator)
    v = torch.randn(n, inner_dim, generator=generator)
    u[:rank, :rank] += 2.0 * torch.eye(rank)
    return u, v


@pytest.mark.parametrize("shape,rank", [((6, 5, 4), 3), ((8, 7, 5), 2), ((5, 4, 3), 3)])
def test_make_gar_reconstructs_truncated_factorization(shape, rank):
    """The compact GAR factors should reconstruct the rank-truncated product exactly."""
    m, n, inner_dim = shape
    u, v = _make_factor_with_invertible_leading_block(m, n, inner_dim, rank)

    result = make_gar(u, v, rank)

    left_factor = _reconstruct_left_factor(result, rank)
    reconstructed = left_factor @ result.gar.v.T
    expected = u[:, :rank] @ v[:, :rank].T

    assert result.gar.u.shape == (m - rank, rank)
    assert result.gar.v.shape == (n, rank)
    assert result.g.shape == (rank, rank)
    assert torch.allclose(reconstructed, expected, atol=1e-5, rtol=1e-5)


def test_make_gar_normalizes_leading_block_to_identity():
    """The leading block should become the identity after the gauge transform."""
    u = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 3.0],
            [1.0, 1.0, 1.0],
        ]
    )
    v = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )

    result = make_gar(u, v, 3)
    u_tilde = u[:, :3] @ result.g

    assert torch.allclose(u_tilde[:3, :], torch.eye(3), atol=1e-6, rtol=1e-6)
    assert torch.allclose(u_tilde[3:, :], result.gar.u, atol=1e-6, rtol=1e-6)
    assert torch.allclose(result.g, torch.linalg.inv(u[:3, :3]), atol=1e-6, rtol=1e-6)


def test_make_gar_logs_and_raises_when_leading_block_is_singular(caplog):
    """GAR should fail if the leading pivot block is singular under the stricter contract."""
    u = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    v = torch.randn(5, 3)

    logger = _GAR_MODULE.log
    old_propagate = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level("ERROR", logger=logger.name):
            with pytest.raises(ValueError, match="leading 2x2 block is not invertible"):
                make_gar(u, v, 2)
    finally:
        logger.propagate = old_propagate

    assert "leading 2x2 block is not invertible" in caplog.text


def test_make_gar_can_fail_even_if_later_rows_would_work():
    """The stricter contract rejects matrices whose leading block is singular."""
    u = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    v = torch.randn(5, 3)

    with pytest.raises(ValueError, match="leading 2x2 block is not invertible"):
        make_gar(u, v, 2)


@pytest.mark.parametrize(
    "u_shape,v_shape,rank,error_match",
    [
        ((4, 3), (5, 2), 2, "same inner dimension"),
        ((4, 3), (5, 3), 0, "must be positive"),
        ((4, 3), (5, 3), 4, "inner dimension"),
        ((2, 3), (5, 3), 3, "number of rows"),
    ],
)
def test_make_gar_validates_inputs(u_shape, v_shape, rank, error_match):
    """Input validation should reject inconsistent shapes and invalid target ranks."""
    u = torch.randn(*u_shape)
    v = torch.randn(*v_shape)

    with pytest.raises(ValueError, match=error_match):
        make_gar(u, v, rank)


def test_prune_weights_with_gar_switches_impl_and_preserves_forward():
    """GAR pruning should compact the factors and keep the same rank-truncated map."""
    layer = ODLinear(4, 6, bias=True, frwd_impl=ODImpl.SLICED, dtype=torch.float32)
    layer.inner_dim = 3

    layer.weight_u.data = torch.tensor(
        [
            [1.0, -1.0, 0.5, 2.0],
            [0.0, 1.5, -0.5, 1.0],
            [2.0, 0.0, 1.0, -1.0],
            [7.0, 7.0, 7.0, 7.0],
        ]
    )
    layer.weight_v.data = torch.tensor(
        [
            [2.0, 0.0, 0.0, 5.0],
            [0.0, 3.0, 0.0, 6.0],
            [0.0, 0.0, 4.0, 7.0],
            [1.0, -1.0, 2.0, 8.0],
            [0.5, 1.0, -0.5, 9.0],
            [-1.5, 0.0, 1.0, 10.0],
        ]
    )
    layer.bias.data = torch.tensor([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])

    inputs = torch.tensor(
        [
            [1.0, 0.5, -1.0, 2.0],
            [-0.5, 1.5, 0.0, -2.0],
        ]
    )

    expected_weight = layer.get_weight().clone()
    expected_output = layer(inputs).clone()

    layer.prune_weights(use_gar=True)

    assert layer.impl == ODImpl.GAR
    assert layer.weight_u.shape == (3, 4)
    assert layer.weight_v.shape == (3, 3)
    assert torch.allclose(layer.get_weight(), expected_weight, atol=1e-6, rtol=1e-6)
    assert torch.allclose(layer(inputs), expected_output, atol=1e-6, rtol=1e-6)


def test_prune_weights_raises_if_called_again_after_gar_conversion():
    """GAR-pruned layers should reject any further pruning pass."""
    layer = ODLinear(4, 6, bias=False, frwd_impl=ODImpl.SLICED, dtype=torch.float32)
    layer.inner_dim = 3

    layer.weight_u.data = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [9.0, 9.0, 9.0, 9.0],
        ]
    )
    layer.weight_v.data = torch.tensor(
        [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 3.0, 0.0, 1.0],
            [0.0, 0.0, 4.0, 1.0],
            [1.0, -1.0, 2.0, 1.0],
            [0.5, 1.0, -0.5, 1.0],
            [-1.5, 0.0, 1.0, 1.0],
        ]
    )

    layer.prune_weights(use_gar=True)

    with pytest.raises(RuntimeError, match="converted to GAR mode"):
        layer.prune_weights()


def test_inner_dim_cannot_change_after_gar_conversion():
    """GAR-pruned layers should reject post-conversion rank changes."""
    layer = ODLinear(4, 6, bias=False, frwd_impl=ODImpl.SLICED, dtype=torch.float32)
    layer.inner_dim = 3

    layer.weight_u.data = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [9.0, 9.0, 9.0, 9.0],
        ]
    )
    layer.weight_v.data = torch.tensor(
        [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 3.0, 0.0, 1.0],
            [0.0, 0.0, 4.0, 1.0],
            [1.0, -1.0, 2.0, 1.0],
            [0.5, 1.0, -0.5, 1.0],
            [-1.5, 0.0, 1.0, 1.0],
        ]
    )

    layer.prune_weights(use_gar=True)

    with pytest.raises(RuntimeError, match="Cannot modify `inner_dim`"):
        layer.inner_dim = 2


def _format_optional_ms(seconds):
    if seconds is None:
        return "n/a"
    return f"{seconds * 1e3:.3f} ms"


def _measure_compiled_gar_subprocess(
    in_features: int,
    out_features: int,
    rank: int,
    batch_size: int,
) -> tuple[dict[str, float], str | None]:
    """Run torch.compile timings out of process so Inductor crashes cannot kill pytest."""
    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.2")
    env["PATH"] = f"/usr/local/cuda-12.2/bin:{env.get('PATH', '')}"

    compiled_times = {}
    compile_errors = []
    for case in ("full_dense", "dense", "sliced", "gar"):
        script = f"""
import json
from time import perf_counter
import torch
from flexrank.layers.base import ODImpl
from flexrank.layers.decomposition import decompose_linear
from flexrank.layers.linear import ODLinear

torch.manual_seed(0)
device = torch.device("cuda")
in_features = {in_features}
out_features = {out_features}
rank = {rank}
batch_size = {batch_size}
warmup_steps = 20
timed_steps = 200
dtype = torch.float32
case = {case!r}

def measure(layer, inputs):
    with torch.no_grad():
        for _ in range(warmup_steps):
            layer(inputs)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        for _ in range(timed_steps):
            layer(inputs)
        end_event.record()
        torch.cuda.synchronize(device)
        return start_event.elapsed_time(end_event) / timed_steps / 1e3

linear_layer = torch.nn.Linear(
    in_features,
    out_features,
    bias=True,
    dtype=dtype,
    device=device,
).to(device)
base_layer = decompose_linear(linear_layer, device=device, dtype=dtype)
base_layer.inner_dim = rank

if case == "full_dense":
    layer = torch.nn.Linear(in_features, out_features, bias=True, dtype=dtype, device=device).to(device)
    layer.weight.data.copy_(base_layer.get_weight().contiguous())
    layer.bias.data.copy_(base_layer.bias.data)
else:
    impl = ODImpl.SLICED if case == "sliced" else ODImpl.DENSE
    use_gar = case == "gar"
    layer = ODLinear(
        in_features,
        out_features,
        bias=True,
        frwd_impl=impl,
        dtype=dtype,
        device=device,
    ).to(device)
    layer.load_state_dict(base_layer.state_dict())
    layer.inner_dim = rank
    layer.prune_weights(use_gar=use_gar)

inputs = torch.randn(batch_size, in_features, device=device)
compiled_layer = torch.compile(layer)
print(json.dumps({{case: measure(compiled_layer, inputs)}}))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            stderr_tail = result.stderr.strip().splitlines()[-1:] or ["no stderr"]
            compile_errors.append(
                f"{case}: torch.compile subprocess failed with code "
                f"{result.returncode}: {stderr_tail[0]}"
            )
            continue

        try:
            compiled_times.update(json.loads(result.stdout.strip().splitlines()[-1]))
        except (IndexError, json.JSONDecodeError) as err:
            compile_errors.append(
                f"{case}: torch.compile subprocess returned unparsable "
                f"output: {type(err).__name__}: {err}"
            )

    return compiled_times, " | ".join(compile_errors) if compile_errors else None


@pytest.mark.skipif(
    os.environ.get("RUN_PERF_TESTS") != "1",
    reason="Set RUN_PERF_TESTS=1 to run timing benchmarks.",
)
def test_forward_timing_gar_vs_dense():  # pylint: disable=too-many-locals,too-many-statements
    """Measure forward-pass wall clock time for dense, sliced, and GAR layouts."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_features = 4096
    out_features = 4096
    rank = 3840
    batch_size = 1024
    warmup_steps = 20
    timed_steps = 200

    linear_layer = torch.nn.Linear(
        in_features,
        out_features,
        bias=True,
        dtype=torch.float32,
        device=device,
    ).to(device)
    base_layer = decompose_linear(linear_layer, device=device, dtype=torch.float32)
    base_layer.inner_dim = rank

    dense_layer = ODLinear(
        in_features,
        out_features,
        bias=True,
        frwd_impl=ODImpl.DENSE,
        dtype=torch.float32,
        device=device,
    ).to(device)
    dense_layer.load_state_dict(base_layer.state_dict())
    dense_layer.inner_dim = rank
    dense_layer.prune_weights(use_gar=False)

    sliced_layer = ODLinear(
        in_features,
        out_features,
        bias=True,
        frwd_impl=ODImpl.SLICED,
        dtype=torch.float32,
        device=device,
    ).to(device)
    sliced_layer.load_state_dict(base_layer.state_dict())
    sliced_layer.inner_dim = rank
    sliced_layer.prune_weights(use_gar=False)

    gar_layer = ODLinear(
        in_features,
        out_features,
        bias=True,
        frwd_impl=ODImpl.DENSE,
        dtype=torch.float32,
        device=device,
    ).to(device)
    gar_layer.load_state_dict(base_layer.state_dict())
    gar_layer.inner_dim = rank
    gar_layer.prune_weights(use_gar=True)

    full_dense_layer = torch.nn.Linear(
        in_features,
        out_features,
        bias=True,
        dtype=torch.float32,
        device=device,
    ).to(device)
    with torch.no_grad():
        full_dense_layer.weight.copy_(dense_layer.get_weight().contiguous())
        full_dense_layer.bias.copy_(dense_layer.bias)

    inputs = torch.randn(batch_size, in_features, device=device)

    with torch.no_grad():
        dense_output = dense_layer(inputs)
        sliced_output = sliced_layer(inputs)
        gar_output = gar_layer(inputs)
        full_dense_output = full_dense_layer(inputs)
    assert torch.allclose(dense_output, sliced_output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dense_output, gar_output, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dense_output, full_dense_output, atol=1e-3, rtol=1e-3)

    def _measure(layer):
        with torch.no_grad():
            for _ in range(warmup_steps):
                layer(inputs)
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                start_event.record()
                for _ in range(timed_steps):
                    layer(inputs)
                end_event.record()
                torch.cuda.synchronize(device)
                return start_event.elapsed_time(end_event) / timed_steps / 1e3

            start = perf_counter()
            for _ in range(timed_steps):
                layer(inputs)
            end = perf_counter()
            return (end - start) / timed_steps

    dense_time = _measure(dense_layer)
    sliced_time = _measure(sliced_layer)
    gar_time = _measure(gar_layer)
    full_dense_time = _measure(full_dense_layer)
    compiled_full_dense_time = None
    compiled_dense_time = None
    compiled_sliced_time = None
    compiled_gar_time = None
    compile_errors = []

    run_compile_timing = os.environ.get("RUN_TORCH_COMPILE_PERF_TESTS") == "1"
    if device.type == "cuda" and run_compile_timing and hasattr(torch, "compile"):
        compiled_times, compile_error = _measure_compiled_gar_subprocess(
            in_features,
            out_features,
            rank,
            batch_size,
        )
        compiled_full_dense_time = compiled_times.get("full_dense")
        compiled_dense_time = compiled_times.get("dense")
        compiled_sliced_time = compiled_times.get("sliced")
        compiled_gar_time = compiled_times.get("gar")
        if compile_error is not None:
            compile_errors.append(compile_error)
    elif device.type == "cuda" and not run_compile_timing:
        compile_errors.append("set RUN_TORCH_COMPILE_PERF_TESTS=1 to include torch.compile timings")

    full_dense_best_time = min(
        t for t in [full_dense_time, compiled_full_dense_time] if t is not None
    )
    dense_best_time = min(t for t in [dense_time, compiled_dense_time] if t is not None)
    sliced_best_time = min(t for t in [sliced_time, compiled_sliced_time] if t is not None)
    gar_best_time = min(t for t in [gar_time, compiled_gar_time] if t is not None)
    ideal_gar_ratio = (2 * out_features - rank) / (2 * out_features)
    ideal_gar_speedup = 1 / ideal_gar_ratio

    print(
        "\nGAR forward timing"
        f"\n  device={device.type}  rank={rank}  batch={batch_size}"
        "\n\n  Timings"
        "\n  -------------------------------"
        f"\n  Full dense      {full_dense_time * 1e3:>9.3f} ms"
        f"\n  Dense OD        {dense_time * 1e3:>9.3f} ms"
        f"\n  Sliced OD       {sliced_time * 1e3:>9.3f} ms"
        f"\n  GAR             {gar_time * 1e3:>9.3f} ms"
        f"\n  compiled Full   {_format_optional_ms(compiled_full_dense_time):>12}"
        f"\n  compiled Dense  {_format_optional_ms(compiled_dense_time):>12}"
        f"\n  compiled Sliced {_format_optional_ms(compiled_sliced_time):>12}"
        f"\n  compiled GAR    {_format_optional_ms(compiled_gar_time):>12}"
        "\n\n  Cost vs Full dense (>1 means slower)"
        "\n  -------------------------------"
        f"\n  Dense OD / Full dense      {dense_time / full_dense_time:>7.3f}x"
        f"\n  Sliced OD / Full dense     {sliced_time / full_dense_time:>7.3f}x"
        f"\n  GAR / Full dense           {gar_time / full_dense_time:>7.3f}x"
        f"\n  best Dense OD / best Full  {dense_best_time / full_dense_best_time:>7.3f}x"
        f"\n  best GAR / best Full       {gar_best_time / full_dense_best_time:>7.3f}x"
        "\n\n  Speedups"
        "\n  -------------------------------"
        f"\n  Sliced OD / Dense OD       {dense_time / sliced_time:>7.3f}x"
        f"\n  GAR / Dense OD             {dense_time / gar_time:>7.3f}x"
        f"\n  GAR / Sliced OD            {sliced_time / gar_time:>7.3f}x"
        f"\n  best GAR / best Dense OD   {dense_best_time / gar_best_time:>7.3f}x"
        f"\n  best GAR / best Sliced OD  {sliced_best_time / gar_best_time:>7.3f}x"
        f"\n  ideal GAR / two-linear     {ideal_gar_speedup:>7.3f}x"
    )
    if compile_errors:
        print("\n  Compile diagnostics")
        print("  -------------------------------")
        for error in compile_errors:
            print(f"  {error}")

    if device.type == "cuda":
        assert gar_best_time < dense_best_time * 0.9
        assert gar_best_time < sliced_best_time * 0.9
