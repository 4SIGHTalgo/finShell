from __future__ import annotations

from pathlib import Path
from typing import Any


def render_label_audit(
    path: Path,
    *,
    real_equity: list[float],
    random_paths: list[list[float]],
    pointwise_p95: list[float],
    class_counts: dict[str, int],
    dpi: int,
) -> None:
    from finshell.plotting import _load_pyplot

    plt = _load_pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    equity_axis, balance_axis = axes
    for random_path in random_paths:
        equity_axis.plot(random_path, color="#94a3b8", alpha=0.14, linewidth=0.8)
    equity_axis.plot(pointwise_p95, color="#dc2626", linestyle="--", linewidth=1.8, label="Random p95")
    equity_axis.plot(real_equity, color="black", linewidth=2.2, label="Favorable-label path")
    equity_axis.axhline(0.0, color="#64748b", linewidth=0.8)
    equity_axis.set_title("Label equity vs same-count null")
    equity_axis.set_xlabel("Development event")
    equity_axis.set_ylabel("Cumulative outcome")
    equity_axis.legend(frameon=False)

    labels = list(class_counts)
    counts = [class_counts[label] for label in labels]
    total = max(1, sum(counts))
    bars = balance_axis.bar(labels, counts, color=["#dc2626", "#94a3b8", "#2563eb"][: len(labels)])
    for bar, count in zip(bars, counts):
        balance_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{count / total:.1%}",
            ha="center",
            va="bottom",
        )
    balance_axis.set_title("Triple-barrier class balance")
    balance_axis.set_xlabel("Label")
    balance_axis.set_ylabel("Rows")
    figure.savefig(path, dpi=dpi, format=path.suffix.lstrip("."))
    plt.close(figure)


def _normal_pdf(x_values: Any, mean: float, std: float) -> Any:
    import numpy as np

    return np.exp(-0.5 * ((x_values - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))


def render_selector_cv(path: Path, *, metrics: Any, dpi: int) -> None:
    import numpy as np

    from finshell.plotting import _load_pyplot

    plt = _load_pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"validate": "#2563eb", "test": "#dc2626"}
    for axis, column, title in (
        (axes[0], "total_return", "Selected economic return"),
        (axes[1], "average_precision", "Average precision"),
    ):
        for partition in ("validate", "test"):
            values = metrics.loc[metrics["partition"].eq(partition), column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            color = colors[partition]
            bins = min(12, max(3, int(np.ceil(np.sqrt(len(values))))))
            axis.hist(values, bins=bins, density=True, color=color, alpha=0.12)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if std > 0.0:
                x_values = np.linspace(mean - 4.0 * std, mean + 4.0 * std, 240)
                axis.plot(
                    x_values,
                    _normal_pdf(x_values, mean, std),
                    color=color,
                    linewidth=2.0,
                    label=f"{partition} fit",
                )
            else:
                axis.axvline(mean, color=color, linewidth=2.0, label=f"{partition} degenerate")
            axis.plot(values, np.zeros_like(values), "|", color=color, alpha=0.8, markersize=8)
        axis.set_title(title)
        axis.set_xlabel(column.replace("_", " "))
        axis.set_ylabel("Density")
        limits = metric_axis_limits(column)
        if limits is not None:
            axis.set_xlim(*limits)
        axis.legend(frameon=False)
    figure.savefig(path, dpi=dpi, format=path.suffix.lstrip("."))
    plt.close(figure)


def metric_axis_limits(column: str) -> tuple[float, float] | None:
    if column in {"average_precision", "roc_auc", "prevalence"}:
        return (0.0, 1.05)
    return None


def render_oos_audit(
    path: Path,
    *,
    selected_equity: list[float],
    random_paths: list[list[float]],
    pointwise_p95: list[float],
    no_selector_equity: list[float],
    dpi: int,
) -> None:
    from finshell.plotting import _load_pyplot

    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for random_path in random_paths:
        axis.plot(random_path, color="#94a3b8", alpha=0.16, linewidth=0.8)
    axis.plot(pointwise_p95, color="#dc2626", linestyle="--", linewidth=1.8, label="Random p95")
    axis.plot(
        no_selector_equity,
        color="#2563eb",
        linestyle="--",
        linewidth=1.8,
        label="No selector",
    )
    axis.plot(selected_equity, color="black", linewidth=2.3, label="Selected OOS")
    axis.axhline(0.0, color="#64748b", linewidth=0.8)
    axis.set_title("OOS selected trades vs controls")
    axis.set_xlabel("Quarantine event")
    axis.set_ylabel("Cumulative outcome")
    axis.legend(frameon=False)
    figure.savefig(path, dpi=dpi, format=path.suffix.lstrip("."))
    plt.close(figure)
