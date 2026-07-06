"""Plotting helpers for FlexTrain training and FlexRank profile metrics."""

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import cycle
from math import ceil
from typing import NamedTuple, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.texmanager import TexManager

_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H"]
_LINESTYLES = ["-"] * len(_MARKERS)

__all__ = [
    "FlexRankProfilePlotConfig",
    "SubmodelPlotConfig",
    "history_plot_data",
    "save_flexrank_profile_plot",
    "save_submodel_plot",
    "set_rcparams",
    "show_flexrank_profile_plot",
    "show_submodel_plot",
    "submodel_plot_data",
]


def _is_latex_available() -> bool:
    if shutil.which("latex") is None:
        return False

    try:
        TexManager().get_text_width_height_descent("lp", 12)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False

    return True


_DEFAULT_RC_PARAMS = {
    "legend.fontsize": 11,
    "text.usetex": _is_latex_available(),
    "font.family": "serif",
    "font.size": 14,
    "figure.dpi": 164,
    "savefig.dpi": 164,
    "savefig.format": "pdf",
    "axes.titlesize": 16,
}


LineData: TypeAlias = (
    Mapping[str, Sequence[float]] | tuple[Sequence[float], Sequence[float]] | Sequence[float]
)
MetricPlotData: TypeAlias = dict[str, dict[str, LineData]]
ProfileMetric: TypeAlias = dict[str, float]
ProfileSeries: TypeAlias = Sequence[tuple[Sequence[int | float], Sequence[ProfileMetric], str]]


@dataclass(frozen=True)
class FlexRankProfilePlotConfig:
    """Configuration for FlexRank profile plots."""

    full_size: int
    full_loss: float
    full_accuracy: float | None = None
    full_label: str = "Original model"
    loss_yscale: str | None = None
    alpha: float = 1.0


@dataclass(frozen=True)
class SubmodelPlotConfig:
    """Configuration for generic submodel metric plots."""

    xlabel: str = "Epochs"
    max_rows: int = 1
    use_markers: bool = False
    save_format: str = "pdf"


class _StyleMaps(NamedTuple):
    colors: dict[str, object]
    markers: dict[str, str]
    linestyles: dict[str, str]


class _BaselineMetric(NamedTuple):
    size: int
    value: float
    label: str


class _FlexRankMetricPlotConfig(NamedTuple):
    key: str
    ylabel: str
    baseline: _BaselineMetric
    yscale: str | None = None
    alpha: float = 1.0


class _SubmodelMetricName(NamedTuple):
    label: str
    yscale: str | None = None


class _SubmodelCanvas(NamedTuple):
    fig: plt.Figure
    axes: np.ndarray
    n_rows: int
    n_cols: int


def _normalize_line(line: LineData) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(line, dict):
        x = line.get("x")
        y = line["y"]
    elif isinstance(line, tuple) and len(line) == 2:
        x, y = line
    else:
        x, y = None, line

    if x is None:
        x = np.arange(len(y))
    return np.asarray(x), np.asarray(y)


def _plot_output_path(output_dir: str, name: str, save_format: str) -> tuple[str, str]:
    output_path = os.fspath(output_dir)
    _, ext = os.path.splitext(output_path)
    if ext and not os.path.isdir(output_path):
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        output_stem, _ = os.path.splitext(output_path)
        return f"{output_stem}.{save_format}", save_format

    os.makedirs(output_path, exist_ok=True)
    filename = name.replace(" ", "_").lower()
    filename_stem, _ = os.path.splitext(filename)
    filename = f"{filename_stem}.{save_format}"
    return os.path.join(output_path, filename), save_format


def _style_maps(
    labels: Sequence[str],
) -> _StyleMaps:
    n_labels = len(labels)
    if n_labels <= 10:
        cmap_name = "tab10"
    elif n_labels <= 20:
        cmap_name = "tab20"
    else:
        cmap_name = "viridis"
    cmap = plt.get_cmap(cmap_name)
    cmap_colors = getattr(cmap, "colors", None)
    if cmap_colors is not None:
        colors = list(cmap_colors[:n_labels])
    elif n_labels > 1:
        colors = [cmap(i / (n_labels - 1)) for i in range(n_labels)]
    else:
        colors = [cmap(0.5)]

    return _StyleMaps(
        colors=dict(zip(labels, colors)),
        markers=dict(zip(labels, cycle(_MARKERS))),
        linestyles=dict(zip(labels, cycle(_LINESTYLES))),
    )


def _submodel_grid_shape(n_metrics: int, max_rows: int) -> tuple[int, int]:
    n_rows = min(max_rows, n_metrics)
    n_cols = int(ceil(n_metrics / n_rows))
    return n_rows, n_cols


def _ordered_line_labels(data: MetricPlotData) -> list[str]:
    seen = set()
    labels = []
    for lines_data in data.values():
        for label in lines_data:
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def _metric_name(metric: str) -> _SubmodelMetricName:
    if "|" not in metric:
        return _SubmodelMetricName(metric)
    label, yscale = metric.split("|", maxsplit=1)
    return _SubmodelMetricName(label, yscale)


def _subplot_position(index: int, n_rows: int) -> tuple[int, int]:
    return index % n_rows, index // n_rows


def _submodel_marker_map(
    labels: Sequence[str],
    use_markers: bool,
) -> dict[str, str | None]:
    return {
        label: marker if use_markers else None for label, marker in zip(labels, cycle(_MARKERS))
    }


def _make_submodel_canvas(n_metrics: int, max_rows: int) -> _SubmodelCanvas:
    n_rows, n_cols = _submodel_grid_shape(n_metrics, max_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2 * n_rows))
    return _SubmodelCanvas(
        fig=fig,
        axes=np.array(axes, dtype=object).reshape(n_rows, n_cols),
        n_rows=n_rows,
        n_cols=n_cols,
    )


def _plot_submodel_metric_lines(
    ax: Axes,
    lines_data: dict[str, LineData],
    styles: _StyleMaps,
    markers: Mapping[str, str | None],
) -> None:
    for label, line in lines_data.items():
        x, y = _normalize_line(line)
        ax.plot(
            x,
            y,
            label=label,
            marker=markers.get(label),
            color=styles.colors.get(label),
        )


def _configure_submodel_axis(
    ax: Axes,
    metric: _SubmodelMetricName,
    xlabel: str,
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.label)
    ax.grid(alpha=0.5, linestyle=":")
    if metric.yscale:
        ax.set_yscale(metric.yscale)
    ax.legend(loc="upper right")


def _plot_submodel_metrics(
    canvas: _SubmodelCanvas,
    data: MetricPlotData,
    xlabel: str,
    styles: _StyleMaps,
    markers: Mapping[str, str | None],
) -> None:
    for index, (metric, lines_data) in enumerate(data.items()):
        row, col = _subplot_position(index, canvas.n_rows)
        ax = canvas.axes[row, col]
        _plot_submodel_metric_lines(ax, lines_data, styles, markers)
        _configure_submodel_axis(ax, _metric_name(metric), xlabel)


def _hide_unused_axes(
    axes: np.ndarray,
    n_metrics: int,
    n_rows: int,
    n_cols: int,
) -> None:
    for index in range(n_metrics, n_rows * n_cols):
        row, col = _subplot_position(index, n_rows)
        axes[row, col].axis("off")


def _make_submodel_plot(
    data: MetricPlotData,
    xlabel: str = "Epochs",
    max_rows: int = 1,
    use_markers: bool = False,
) -> plt.Figure:
    n_metrics = len(data)
    if not n_metrics:
        raise ValueError("Data must contain at least one metric.")

    canvas = _make_submodel_canvas(n_metrics, max_rows)
    ordered_labels = _ordered_line_labels(data)
    styles = _style_maps(ordered_labels)
    markers = _submodel_marker_map(ordered_labels, use_markers)

    _plot_submodel_metrics(canvas, data, xlabel, styles, markers)
    _hide_unused_axes(canvas.axes, n_metrics, canvas.n_rows, canvas.n_cols)

    return canvas.fig


def _plot_flexrank_metric(
    ax,
    series: ProfileSeries,
    config: _FlexRankMetricPlotConfig,
    styles: _StyleMaps,
) -> None:
    for x, metrics, label in series:
        ax.plot(
            x,
            [item[config.key] for item in metrics],
            label=label,
            color=styles.colors[label],
            marker=styles.markers[label],
            linestyle=styles.linestyles[label],
            linewidth=1.8,
            markersize=5.5,
            alpha=config.alpha,
        )

    ax.scatter(
        [config.baseline.size],
        [config.baseline.value],
        label=config.baseline.label,
        color="firebrick",
        marker="*",
        s=90,
        zorder=5,
        alpha=config.alpha,
    )
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel(config.ylabel)
    ax.grid(alpha=0.5, linestyle=":")
    if config.yscale is not None:
        ax.set_yscale(config.yscale)
    ax.legend(loc="upper right")


def show_flexrank_profile_plot(
    name: str,
    series: ProfileSeries,
    config: FlexRankProfilePlotConfig,
) -> None:
    """Display loss and optional accuracy over FlexRank submodel profiles."""
    _make_flexrank_profile_plot(name, series, config)
    plt.show()


def _make_flexrank_profile_plot(
    name: str,
    series: ProfileSeries,
    config: FlexRankProfilePlotConfig,
) -> plt.Figure:
    labels = [label for _, _, label in series]
    styles = _style_maps(labels)
    has_accuracy = any(metrics and "eval_accuracy" in metrics[0] for _, metrics, _ in series)

    if has_accuracy:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes = np.asarray(axes, dtype=object)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        axes = np.asarray([ax], dtype=object)

    _plot_flexrank_metric(
        axes[0],
        series,
        _FlexRankMetricPlotConfig(
            key="eval_loss",
            ylabel="Loss",
            baseline=_BaselineMetric(
                config.full_size,
                config.full_loss,
                config.full_label,
            ),
            yscale=config.loss_yscale,
            alpha=config.alpha,
        ),
        styles,
    )

    if has_accuracy:
        if config.full_accuracy is None:
            raise ValueError("full_accuracy is required for accuracy plots.")
        _plot_flexrank_metric(
            axes[1],
            series,
            _FlexRankMetricPlotConfig(
                key="eval_accuracy",
                ylabel="Accuracy",
                baseline=_BaselineMetric(
                    config.full_size,
                    config.full_accuracy,
                    config.full_label,
                ),
                alpha=config.alpha,
            ),
            styles,
        )

    fig.suptitle(name)
    fig.tight_layout()
    return fig


def save_flexrank_profile_plot(
    output_dir: str,
    name: str,
    series: ProfileSeries,
    config: FlexRankProfilePlotConfig,
    save_format: str = "pdf",
) -> None:
    """Save loss and optional accuracy over FlexRank submodel profiles."""
    path, output_format = _plot_output_path(output_dir, name, save_format)
    fig = _make_flexrank_profile_plot(name, series, config)
    fig.savefig(path, format=output_format)
    plt.close(fig)


def show_submodel_plot(
    name: str,
    data: MetricPlotData,
    config: SubmodelPlotConfig | None = None,
) -> None:
    """Display one or more metric panels for labeled submodel series."""
    config = config or SubmodelPlotConfig()
    fig = _make_submodel_plot(
        data,
        config.xlabel,
        config.max_rows,
        config.use_markers,
    )
    fig.suptitle(name)
    fig.tight_layout()
    plt.show()


def save_submodel_plot(
    output_dir: str,
    name: str,
    data: MetricPlotData,
    config: SubmodelPlotConfig | None = None,
) -> None:
    """Save one or more metric panels for labeled submodel series."""
    config = config or SubmodelPlotConfig()
    path, output_format = _plot_output_path(output_dir, name, config.save_format)
    fig = _make_submodel_plot(
        data,
        config.xlabel,
        config.max_rows,
        config.use_markers,
    )
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, format=output_format)
    plt.close(fig)


def history_plot_data(
    metrics: dict,
    *,
    train_loss_key: str = "train_loss_history",
    eval_loss_key: str = "eval_loss_history",
    train_acc_key: str = "train_accuracy_history",
    eval_acc_key: str = "eval_accuracy_history",
) -> MetricPlotData:
    """Convert training history metrics into generic submodel plot data."""
    data: MetricPlotData = {}
    if train_loss_key in metrics or eval_loss_key in metrics:
        data["Loss"] = {}
        if train_loss_key in metrics:
            data["Loss"]["Train"] = metrics[train_loss_key]
        if eval_loss_key in metrics:
            data["Loss"]["Eval"] = metrics[eval_loss_key]
    if train_acc_key in metrics or eval_acc_key in metrics:
        data["Accuracy"] = {}
        if train_acc_key in metrics:
            data["Accuracy"]["Train"] = metrics[train_acc_key]
        if eval_acc_key in metrics:
            data["Accuracy"]["Eval"] = metrics[eval_acc_key]
    return data


def submodel_plot_data(
    series: ProfileSeries,
    config: FlexRankProfilePlotConfig,
) -> MetricPlotData:
    """Convert FlexRank profile metrics into generic submodel plot data."""
    loss_metric = "Loss"
    if config.loss_yscale is not None:
        loss_metric = f"{loss_metric}|{config.loss_yscale}"

    data: MetricPlotData = {
        loss_metric: {label: (x, [item["eval_loss"] for item in y]) for x, y, label in series}
    }
    data[loss_metric][config.full_label] = ([config.full_size], [config.full_loss])

    has_accuracy = any(y and "eval_accuracy" in y[0] for _, y, _ in series)
    if has_accuracy and config.full_accuracy is not None:
        data["Accuracy"] = {
            label: (x, [item["eval_accuracy"] for item in y]) for x, y, label in series
        }
        data["Accuracy"][config.full_label] = (
            [config.full_size],
            [config.full_accuracy],
        )

    return data


def set_rcparams(params: dict):
    """Update Matplotlib runtime configuration for plots from this module."""
    plt.rcParams.update(params)


set_rcparams(_DEFAULT_RC_PARAMS)
