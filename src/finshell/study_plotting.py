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
