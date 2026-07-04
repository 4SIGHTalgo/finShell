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


def render_selector_cv(path: Path, *, distributions: dict[str, Any], dpi: int) -> None:
    import numpy as np

    from finshell.plotting import _load_pyplot

    plt = _load_pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"validate": "#2563eb", "test": "#dc2626"}
    for partition in ("validate", "test"):
        values = np.asarray(distributions["outer"][partition], dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        color = colors[partition]
        bins = min(12, max(3, int(np.ceil(np.sqrt(len(values))))))
        axes[0].hist(values, bins=bins, density=True, color=color, alpha=0.12)
        _plot_normal_fit(axes[0], values, color=color, label=partition)
    axes[0].set_title("Distribution of CPCV fold means")
    axes[0].set_xlabel("fold mean total return")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)

    fold_records = distributions["folds"]
    all_values = [
        value
        for fold in fold_records
        for partition in ("validate", "test")
        for value in fold[partition]
        if np.isfinite(value)
    ]
    if all_values:
        low, high = float(np.min(all_values)), float(np.max(all_values))
        padding = max((high - low) * 0.12, 1e-6)
        x_values = np.linspace(low - padding, high + padding, 320)
        for row, fold in enumerate(fold_records):
            baseline = float(len(fold_records) - row - 1)
            axes[1].axhline(baseline, color="#cbd5e1", linewidth=0.7, zorder=0)
            for partition, direction in (("validate", 1.0), ("test", -1.0)):
                values = np.asarray(fold[partition], dtype=float)
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                density = _scaled_normal_density(x_values, values, height=0.34)
                curve = baseline + direction * density
                axes[1].plot(x_values, curve, color=colors[partition], linewidth=1.5)
                axes[1].fill_between(
                    x_values,
                    baseline,
                    curve,
                    color=colors[partition],
                    alpha=0.13,
                )
                axes[1].scatter(
                    values,
                    np.full(len(values), baseline + direction * 0.04),
                    marker="|",
                    color=colors[partition],
                    s=30,
                    zorder=3,
                )
    labels = [
        f"{fold['fold']}  V{','.join(map(str, fold['validate_group_indices']))} / "
        f"T{','.join(map(str, fold['test_group_indices']))}"
        for fold in fold_records
    ]
    positions = list(reversed(range(len(fold_records))))
    axes[1].set_yticks(positions, labels)
    axes[1].set_title("Nested block-bootstrap distributions within each CPCV fold")
    axes[1].set_xlabel("permutation total return")
    axes[1].set_ylabel("CPCV permutation")
    axes[1].plot([], [], color=colors["validate"], label="validate")
    axes[1].plot([], [], color=colors["test"], label="test")
    axes[1].legend(frameon=False, fontsize="small")
    figure.savefig(path, dpi=dpi, format=path.suffix.lstrip("."))
    plt.close(figure)


def _plot_normal_fit(axis: Any, values: Any, *, color: Any, label: str) -> None:
    import numpy as np

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if std > 0.0:
        x_values = np.linspace(mean - 4.0 * std, mean + 4.0 * std, 240)
        axis.plot(
            x_values,
            _normal_pdf(x_values, mean, std),
            color=color,
            linewidth=2.0,
            label=f"{label} fit",
        )
    else:
        axis.axvline(mean, color=color, linewidth=2.0, label=f"{label} degenerate")
    axis.plot(values, np.zeros_like(values), "|", color=color, alpha=0.8, markersize=8)


def _scaled_normal_density(x_values: Any, values: Any, *, height: float) -> Any:
    import numpy as np

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if std <= 0.0:
        width = max(float(np.ptp(x_values)) * 0.012, 1e-9)
        std = width
    density = _normal_pdf(x_values, mean, std)
    peak = float(np.max(density))
    return density / peak * float(height) if peak > 0.0 else np.zeros_like(x_values)


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


def render_economic_validation(
    path: Path,
    *,
    account_balance_paths: list[list[float]],
    resolution_states: list[str],
    resolution_trades: list[int],
    median_path: list[float],
    upper_balance: float,
    lower_balance: float,
    max_trades: int,
    upper_hit_probability: float,
    lower_hit_probability: float,
    vertical_probability: float,
    median_resolution_trades: float,
    dpi: int,
) -> None:
    from finshell.plotting import _load_pyplot

    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    colors = {"upper": "#16a34a", "lower": "#dc2626", "vertical": "#94a3b8"}
    for simulated, state, resolved_at in zip(
        account_balance_paths,
        resolution_states,
        resolution_trades,
    ):
        end = int(resolved_at) + 1
        axis.plot(
            range(end),
            simulated[:end],
            color=colors[state],
            alpha=0.18,
            linewidth=0.9,
        )
    axis.plot(median_path, color="black", linewidth=2.5, label="Median path")
    axis.axhline(upper_balance, color="#16a34a", linestyle="--", linewidth=1.8, label="Upper balance")
    axis.axhline(lower_balance, color="#dc2626", linestyle="--", linewidth=1.8, label="Lower balance")
    axis.axvline(max_trades, color="#7c3aed", linestyle=":", linewidth=2.0, label="Trade horizon")
    axis.text(
        0.02,
        0.97,
        f"Upper hit: {upper_hit_probability:.1%}   "
        f"Lower hit: {lower_hit_probability:.1%}   "
        f"Vertical: {vertical_probability:.1%}\n"
        f"Median resolution: {median_resolution_trades:.1f} trades",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    axis.set_title("OOS account-balance block bootstrap")
    axis.set_xlabel("Selected OOS trade")
    axis.set_ylabel("Account balance")
    axis.legend(frameon=False, loc="lower right")
    figure.savefig(path, dpi=dpi, format=path.suffix.lstrip("."))
    plt.close(figure)
