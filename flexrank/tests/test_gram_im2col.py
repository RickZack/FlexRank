"""Regression tests for Gram input preprocessing helpers."""

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_im2col():
    """Load `gram.py` directly to avoid package import cycles in tests."""
    gram_path = Path(__file__).resolve().parents[1] / "src" / "flexrank" / "layers" / "gram.py"
    spec = importlib.util.spec_from_file_location("test_gram_module", gram_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_GRAM_MODULE = _load_im2col()
# We load the module by path so we can test internal helpers without importing
# the full package graph and triggering the known circular import chain.
_im2col = getattr(_GRAM_MODULE, "_im2col")
_prepare_layer_input_matrix = getattr(_GRAM_MODULE, "_prepare_layer_input_matrix")
collect_model_grams = _GRAM_MODULE.collect_model_grams


def _legacy_im2col(input_tensor, kernel_size, stride):  # pylint: disable=too-many-locals
    """Reference implementation kept to validate the refactor to `F.unfold`."""
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride

    out_height = (height - kernel_height) // stride_height + 1
    out_width = (width - kernel_width) // stride_width + 1

    col = torch.zeros(
        size=(
            batch_size,
            in_channels,
            kernel_height,
            kernel_width,
            out_height,
            out_width,
        ),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

    for y in range(kernel_height):
        y_max = y + stride_height * out_height
        for x in range(kernel_width):
            x_max = x + stride_width * out_width
            col[:, :, y, x, :, :] = input_tensor[
                :,
                :,
                y:y_max:stride_height,
                x:x_max:stride_width,
            ]

    col = col.permute(0, 4, 5, 1, 2, 3).contiguous()
    return col.view(batch_size * out_height * out_width, -1)


@pytest.mark.parametrize(
    "shape,kernel_size,stride",
    [
        ((2, 3, 5, 7), (3, 3), (1, 1)),
        ((1, 1, 6, 6), (2, 4), (2, 1)),
        ((3, 2, 9, 8), (3, 2), (2, 3)),
        ((1, 4, 4, 4), (1, 1), (1, 1)),
    ],
)
def test_im2col_matches_legacy_implementation(shape, kernel_size, stride):
    """The `F.unfold` implementation should match the legacy loop exactly."""
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(*shape, generator=generator)

    legacy = _legacy_im2col(inputs, kernel_size, stride)
    current = _im2col(inputs, kernel_size, stride, inputs.device)

    assert torch.equal(current, legacy)


class _SequenceModel(torch.nn.Module):
    """Simple module exposing a sequence-shaped linear layer."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3, bias=False)

    def forward(self, inputs):
        """Apply the test projection layer."""
        return self.proj(inputs)


class _ConvModel(torch.nn.Module):
    """Simple module exposing a convolutional layer."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, kernel_size=(2, 2), stride=(1, 1), bias=False)

    def forward(self, inputs):
        """Apply the test convolution layer."""
        return self.conv(inputs)


class _BatchEvaluator:  # pylint: disable=too-few-public-methods
    """Minimal evaluator that forwards a fixed list of batches."""

    def __init__(self, model, batches):
        self.model = model
        self._batches = batches

    def evaluate(self):
        """Run one evaluation pass over the stored batches."""
        self.model.eval()
        with torch.no_grad():
            for batch in self._batches:
                self.model(batch)


def _expected_gram(layer, inputs):
    """Compute the expected normalized Gram matrix for a batch of inputs."""
    matrix = _prepare_layer_input_matrix(layer, inputs)
    return matrix.T @ matrix / matrix.shape[0]


def test_collect_model_grams_caps_sequence_examples_not_rows():
    """`max_data_count` should cap sequence batches by examples, not tokens."""
    model = _SequenceModel()
    batches = [
        torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4),
        torch.arange(2 * 3 * 4, 2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4),
    ]
    evaluator = _BatchEvaluator(model, batches)

    grams = collect_model_grams(
        [("proj", model.proj)],
        evaluator,
        max_data_count=3,
    )

    expected_inputs = torch.cat([batches[0], batches[1][:1]], dim=0)
    expected_gram = _expected_gram(model.proj, expected_inputs)

    assert torch.allclose(grams["proj"], expected_gram)


def test_collect_model_grams_caps_conv_examples_not_patches():
    """`max_data_count` should cap conv batches by examples, not patch rows."""
    model = _ConvModel()
    batches = [
        torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4),
        torch.arange(2 * 1 * 4 * 4, 2 * 2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4),
    ]
    evaluator = _BatchEvaluator(model, batches)

    grams = collect_model_grams(
        [("conv", model.conv)],
        evaluator,
        max_data_count=3,
    )

    expected_inputs = torch.cat([batches[0], batches[1][:1]], dim=0)
    expected_gram = _expected_gram(model.conv, expected_inputs)

    assert torch.allclose(grams["conv"], expected_gram)
