from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finshell.core import ComponentResult, PipelineComponent, PipelineContext
from finshell.cpcv import CPCVFold
from finshell.ingestion import ColumnRoleMap


@dataclass(frozen=True, slots=True)
class PlotConfig:
    enabled: bool = False
    output_subdir: str = "plots"
    image_format: str = "png"
    dpi: int = 150
    max_paths: int = 100
    bootstrap_block_bars: int = 8
    include_fold_bootstrap: bool = True
    strict: bool = False
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not self.output_subdir.strip():
            raise ValueError("output_subdir must not be empty")
        if self.image_format.lower() not in {"png", "svg", "pdf"}:
            raise ValueError("image_format must be one of: png, svg, pdf")
        if self.dpi < 1:
            raise ValueError("dpi must be >= 1")
        if self.max_paths < 1:
            raise ValueError("max_paths must be >= 1")
        if self.bootstrap_block_bars < 1:
            raise ValueError("bootstrap_block_bars must be >= 1")


class LabelDiagnosticsPlot(PipelineComponent):
    def __init__(
        self,
        config: PlotConfig | None = None,
        *,
        name: str = "label_diagnostics_plot",
        data_key: str = "data",
    ) -> None:
        super().__init__(name=name)
        self.config = config or PlotConfig()
        self.data_key = data_key

    def run(self, context: PipelineContext) -> ComponentResult:
        if not self.config.enabled:
            return _skipped(self.name, "plotting_disabled")
        frame, roles = _frame_and_roles(context, self.data_key)
        if not roles.label:
            return self._unavailable("missing_label_role")

        labels = frame[roles.label].dropna()
        class_counts = {
            _value_key(key): int(value)
            for key, value in labels.value_counts().sort_index().items()
        }
        outcomes = _finite_outcomes(frame, roles)
        real_equity: list[float] = []
        random_paths: list[list[float]] = []
        if outcomes.size:
            real_equity = _rounded_cumsum(outcomes)
            rng = np.random.default_rng(self.config.random_seed)
            random_paths = [
                _rounded_cumsum(rng.permutation(outcomes))
                for _ in range(self.config.max_paths)
            ]

        diagnostics: dict[str, Any] = {
            "class_counts": class_counts,
            "real_equity": real_equity,
            "random_equity_paths": random_paths,
            "random_seed": int(self.config.random_seed),
            "sampling_rule": "outcome_permutation",
        }
        context.state["label_plot_diagnostics"] = diagnostics
        figure_path = _figure_path(context, self.config, "label_diagnostics")
        self._render(
            figure_path=figure_path,
            class_counts=class_counts,
            real_equity=real_equity,
            random_paths=random_paths,
        )
        return ComponentResult(
            component=self.name,
            passed=True,
            summary={
                "skipped": False,
                "class_counts": class_counts,
                "random_paths": len(random_paths),
            },
            artifacts={"figure": figure_path},
        )

    def _unavailable(self, reason: str) -> ComponentResult:
        if self.config.strict:
            return ComponentResult(component=self.name, passed=False, summary={"fail_reasons": [reason]})
        return _skipped(self.name, reason)

    def _render(
        self,
        *,
        figure_path: Path,
        class_counts: dict[str, int],
        real_equity: list[float],
        random_paths: list[list[float]],
    ) -> None:
        plt = _load_pyplot()
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        equity_axis, balance_axis = axes
        for path in random_paths:
            equity_axis.plot(path, color="#9ca3af", alpha=0.16, linewidth=0.8)
        if real_equity:
            equity_axis.plot(real_equity, color="black", linewidth=2.0, label="Real path")
            equity_axis.legend(frameon=False)
        else:
            equity_axis.text(0.5, 0.5, "Outcome role unavailable", ha="center", va="center")
        equity_axis.axhline(0.0, color="#6b7280", linewidth=0.8)
        equity_axis.set_title("Outcome equity path")
        equity_axis.set_xlabel("Observation")
        equity_axis.set_ylabel("Cumulative outcome")

        keys = list(class_counts)
        values = [class_counts[key] for key in keys]
        balance_axis.bar(keys, values, color="#2563eb")
        balance_axis.set_title("Class balance")
        balance_axis.set_xlabel("Label")
        balance_axis.set_ylabel("Rows")
        figure.savefig(figure_path, dpi=self.config.dpi, format=self.config.image_format.lower())
        plt.close(figure)


class CPCVDiagnosticsPlot(PipelineComponent):
    def __init__(
        self,
        config: PlotConfig | None = None,
        *,
        name: str = "cpcv_diagnostics_plot",
        data_key: str = "development_data",
    ) -> None:
        super().__init__(name=name)
        self.config = config or PlotConfig()
        self.data_key = data_key

    def run(self, context: PipelineContext) -> ComponentResult:
        if not self.config.enabled:
            return _skipped(self.name, "plotting_disabled")
        frame, roles = _frame_and_roles(context, self.data_key)
        if not roles.outcome or roles.outcome not in frame.columns:
            return self._unavailable("missing_outcome_role")
        folds = context.state.get("cpcv_folds")
        if not isinstance(folds, list) or not folds or not all(isinstance(fold, CPCVFold) for fold in folds):
            return self._unavailable("missing_cpcv_folds")

        outcomes = pd.to_numeric(frame[roles.outcome], errors="coerce")
        plans = context.state.get("bootstrap_plans") if self.config.include_fold_bootstrap else []
        plan_by_fold = {
            getattr(plan, "fold_name", ""): plan
            for plan in plans
        } if isinstance(plans, list) else {}
        fold_diagnostics: list[dict[str, Any]] = []
        for fold in folds:
            validate_total = _partition_total(outcomes, fold.validate_indices)
            test_total = _partition_total(outcomes, fold.test_indices)
            plan = plan_by_fold.get(fold.name)
            bootstrap_totals = [
                _partition_total(outcomes, path.indices)
                for path in getattr(plan, "paths", [])
            ]
            bootstrap_totals = [value for value in bootstrap_totals if np.isfinite(value)]
            fold_diagnostics.append(
                {
                    "fold": fold.name,
                    "validate_total": validate_total,
                    "test_total": test_total,
                    "bootstrap_totals": bootstrap_totals,
                    "bootstrap_fit": _normal_fit(bootstrap_totals),
                }
            )

        validate_totals = [item["validate_total"] for item in fold_diagnostics]
        test_totals = [item["test_total"] for item in fold_diagnostics]
        diagnostics = {
            "validate_totals": validate_totals,
            "test_totals": test_totals,
            "validate_fit": _normal_fit(validate_totals),
            "test_fit": _normal_fit(test_totals),
            "folds": fold_diagnostics,
            "include_fold_bootstrap": bool(self.config.include_fold_bootstrap),
        }
        context.state["cpcv_plot_diagnostics"] = diagnostics
        figure_path = _figure_path(context, self.config, "cpcv_distributions")
        self._render(figure_path, diagnostics)
        return ComponentResult(
            component=self.name,
            passed=True,
            summary={
                "skipped": False,
                "fold_count": len(fold_diagnostics),
                "fold_bootstrap_included": bool(any(item["bootstrap_totals"] for item in fold_diagnostics)),
            },
            artifacts={"figure": figure_path},
        )

    def _unavailable(self, reason: str) -> ComponentResult:
        if self.config.strict:
            return ComponentResult(component=self.name, passed=False, summary={"fail_reasons": [reason]})
        return _skipped(self.name, reason)

    def _render(self, figure_path: Path, diagnostics: dict[str, Any]) -> None:
        plt = _load_pyplot()
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        heldout_axis, bootstrap_axis = axes
        _plot_bell_curve(
            heldout_axis,
            diagnostics["validate_totals"],
            diagnostics["validate_fit"],
            label="Validate",
            color="#2563eb",
        )
        _plot_bell_curve(
            heldout_axis,
            diagnostics["test_totals"],
            diagnostics["test_fit"],
            label="Test",
            color="#dc2626",
        )
        heldout_axis.axvline(0.0, color="#6b7280", linewidth=0.8)
        heldout_axis.set_title("CPCV held-out outcomes")
        heldout_axis.set_xlabel("Aggregate outcome")
        heldout_axis.set_ylabel("Density")
        heldout_axis.legend(frameon=False)

        bootstrap_folds = [item for item in diagnostics["folds"] if item["bootstrap_totals"]]
        if bootstrap_folds:
            colors = plt.get_cmap("tab10").colors
            for index, item in enumerate(bootstrap_folds):
                _plot_bell_curve(
                    bootstrap_axis,
                    item["bootstrap_totals"],
                    item["bootstrap_fit"],
                    label=item["fold"],
                    color=colors[index % len(colors)],
                )
            bootstrap_axis.legend(frameon=False)
        else:
            bootstrap_axis.text(0.5, 0.5, "Fold bootstrap disabled", ha="center", va="center")
        bootstrap_axis.axvline(0.0, color="#6b7280", linewidth=0.8)
        bootstrap_axis.set_title("Per-fold train bootstrap")
        bootstrap_axis.set_xlabel("Aggregate outcome")
        bootstrap_axis.set_ylabel("Density")
        figure.savefig(figure_path, dpi=self.config.dpi, format=self.config.image_format.lower())
        plt.close(figure)


class NullTestDiagnosticsPlot(PipelineComponent):
    def __init__(
        self,
        config: PlotConfig | None = None,
        *,
        name: str = "null_test_diagnostics_plot",
        data_key: str = "development_data",
    ) -> None:
        super().__init__(name=name)
        self.config = config or PlotConfig()
        self.data_key = data_key

    def run(self, context: PipelineContext) -> ComponentResult:
        if not self.config.enabled:
            return _skipped(self.name, "plotting_disabled")
        frame, roles = _frame_and_roles(context, self.data_key)
        if not roles.outcome or roles.outcome not in frame.columns:
            return self._unavailable("missing_outcome_role")
        diagnostics = context.state.get("null_test_diagnostics")
        if not isinstance(diagnostics, dict):
            return self._unavailable("missing_null_test_diagnostics")

        outcomes = pd.to_numeric(frame[roles.outcome], errors="coerce")
        selected_indices = [int(value) for value in diagnostics.get("selected_row_indices", [])]
        sampled_indices = [
            [int(value) for value in path]
            for path in diagnostics.get("random_row_indices", [])[: self.config.max_paths]
        ]
        real_equity = _series_cumsum(outcomes.iloc[selected_indices])
        random_paths = [_series_cumsum(outcomes.iloc[indices]) for indices in sampled_indices]
        plot_diagnostics = {
            "real_equity": real_equity,
            "random_equity_paths": random_paths,
            "selected_row_indices": selected_indices,
            "sampled_row_indices": sampled_indices,
            "random_seed": diagnostics.get("random_seed"),
        }
        context.state["null_plot_diagnostics"] = plot_diagnostics
        figure_path = _figure_path(context, self.config, "null_test_equity_paths")
        self._render(figure_path, real_equity, random_paths)
        return ComponentResult(
            component=self.name,
            passed=True,
            summary={
                "skipped": False,
                "selected_count": len(selected_indices),
                "displayed_random_paths": len(random_paths),
            },
            artifacts={"figure": figure_path},
        )

    def _unavailable(self, reason: str) -> ComponentResult:
        if self.config.strict:
            return ComponentResult(component=self.name, passed=False, summary={"fail_reasons": [reason]})
        return _skipped(self.name, reason)

    def _render(self, figure_path: Path, real_equity: list[float], random_paths: list[list[float]]) -> None:
        plt = _load_pyplot()
        figure, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
        for path in random_paths:
            axis.plot(path, color="#94a3b8", alpha=0.18, linewidth=0.9)
        if real_equity:
            axis.plot(real_equity, color="black", linewidth=2.2, label="Selected trades")
            axis.legend(frameon=False)
        axis.axhline(0.0, color="#6b7280", linewidth=0.8)
        axis.set_title("Selected trades vs same-count null paths")
        axis.set_xlabel("Trade number")
        axis.set_ylabel("Cumulative outcome")
        figure.savefig(figure_path, dpi=self.config.dpi, format=self.config.image_format.lower())
        plt.close(figure)


class TripleBarrierDiagnosticsPlot(PipelineComponent):
    def __init__(
        self,
        config: PlotConfig | None = None,
        *,
        name: str = "triple_barrier_diagnostics_plot",
        result_key: str = "triple_barrier_result",
    ) -> None:
        super().__init__(name=name)
        self.config = config or PlotConfig()
        self.result_key = result_key

    def run(self, context: PipelineContext) -> ComponentResult:
        if not self.config.enabled:
            return _skipped(self.name, "plotting_disabled")
        result = context.state.get(self.result_key)
        if not isinstance(result, pd.DataFrame) or "barrier_return" not in result.columns:
            return self._unavailable("missing_triple_barrier_returns")
        values = pd.to_numeric(result["barrier_return"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            return self._unavailable("no_finite_triple_barrier_returns")

        block_bars = min(int(self.config.bootstrap_block_bars), len(values))
        paths = _block_bootstrap_equity_paths(
            values,
            replicates=int(self.config.max_paths),
            block_bars=block_bars,
            random_seed=int(self.config.random_seed),
        )
        median_path = np.median(np.asarray(paths, dtype=float), axis=0).tolist()
        diagnostics = {
            "bootstrap_equity_paths": paths,
            "median_equity_path": [float(value) for value in median_path],
            "block_bars": block_bars,
            "random_seed": int(self.config.random_seed),
            "sampling_rule": "moving_block_bootstrap",
            "source_return_count": int(len(values)),
        }
        context.state["triple_barrier_plot_diagnostics"] = diagnostics
        figure_path = _figure_path(context, self.config, "triple_barrier_bootstrap_paths")
        self._render(figure_path, paths, median_path)
        return ComponentResult(
            component=self.name,
            passed=True,
            summary={
                "skipped": False,
                "bootstrap_paths": len(paths),
                "block_bars": block_bars,
                "source_return_count": len(values),
            },
            artifacts={"figure": figure_path},
        )

    def _unavailable(self, reason: str) -> ComponentResult:
        if self.config.strict:
            return ComponentResult(component=self.name, passed=False, summary={"fail_reasons": [reason]})
        return _skipped(self.name, reason)

    def _render(self, figure_path: Path, paths: list[list[float]], median_path: list[float]) -> None:
        plt = _load_pyplot()
        figure, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
        for path in paths:
            axis.plot(path, color="#60a5fa", alpha=0.16, linewidth=0.8)
        axis.plot(median_path, color="black", linewidth=2.4, label="Median path")
        axis.axhline(0.0, color="#6b7280", linewidth=0.8)
        axis.set_title("Triple-barrier economic bootstrap paths")
        axis.set_xlabel("Event number")
        axis.set_ylabel("Cumulative barrier return")
        axis.legend(frameon=False)
        figure.savefig(figure_path, dpi=self.config.dpi, format=self.config.image_format.lower())
        plt.close(figure)


def _skipped(component: str, reason: str) -> ComponentResult:
    return ComponentResult(component=component, passed=True, summary={"skipped": True, "reason": reason})


def _frame_and_roles(context: PipelineContext, data_key: str) -> tuple[pd.DataFrame, ColumnRoleMap]:
    frame = context.state.get(data_key)
    roles = context.state.get("roles")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"context.state[{data_key!r}] must contain a pandas DataFrame")
    if not isinstance(roles, ColumnRoleMap):
        raise ValueError("context.state['roles'] must contain a ColumnRoleMap")
    return frame, roles


def _finite_outcomes(frame: pd.DataFrame, roles: ColumnRoleMap) -> np.ndarray:
    if not roles.outcome or roles.outcome not in frame.columns:
        return np.asarray([], dtype=float)
    values = pd.to_numeric(frame[roles.outcome], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _partition_total(outcomes: pd.Series, indices: list[int]) -> float:
    value = outcomes.iloc[indices].sum(min_count=1)
    return float(value) if pd.notna(value) else float("nan")


def _normal_fit(values: list[float]) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"mean": float("nan"), "std": float("nan")}
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return {"mean": float(np.mean(finite)), "std": std}


def _plot_bell_curve(
    axis: Any,
    values: list[float],
    fit: dict[str, float],
    *,
    label: str,
    color: Any,
) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return
    bins = min(12, max(3, int(np.ceil(np.sqrt(finite.size)))))
    axis.hist(finite, bins=bins, density=True, color=color, alpha=0.12)
    mean = float(fit["mean"])
    std = float(fit["std"])
    if np.isfinite(std) and std > 0.0:
        x_values = np.linspace(mean - 4.0 * std, mean + 4.0 * std, 240)
        density = np.exp(-0.5 * ((x_values - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))
        axis.plot(x_values, density, color=color, linewidth=2.0, label=f"{label} normal fit")
    else:
        axis.axvline(mean, color=color, linewidth=2.0, label=f"{label} degenerate fit")
    axis.plot(finite, np.zeros_like(finite), "|", color=color, alpha=0.8, markersize=8)


def _rounded_cumsum(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.round(np.cumsum(values), 12).tolist()]


def _series_cumsum(values: pd.Series) -> list[float]:
    finite = values[np.isfinite(values.to_numpy(dtype=float))].to_numpy(dtype=float)
    return _rounded_cumsum(finite)


def _block_bootstrap_equity_paths(
    values: np.ndarray,
    *,
    replicates: int,
    block_bars: int,
    random_seed: int,
) -> list[list[float]]:
    rng = np.random.default_rng(random_seed)
    paths: list[list[float]] = []
    for _ in range(replicates):
        sampled: list[float] = []
        while len(sampled) < len(values):
            start = int(rng.integers(0, len(values)))
            sampled.extend(values[start : min(len(values), start + block_bars)].tolist())
        path_values = np.asarray(sampled[: len(values)], dtype=float)
        paths.append(_rounded_cumsum(path_values))
    return paths


def _value_key(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def _figure_path(context: PipelineContext, config: PlotConfig, stem: str) -> Path:
    return context.artifact_path(Path(config.output_subdir) / f"{stem}.{config.image_format.lower()}")


def _load_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('plotting requires the optional dependency: pip install "finShell[plots]"') from exc
    return plt
