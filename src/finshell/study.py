from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finshell.core import PipelineContext
from finshell.holdout import HoldoutConfig, HoldoutSplitter
from finshell.ingestion import ColumnRoleMap, DataIngestConfig, DataIngestor
from finshell.label_audit import LabelAuditConfig, LabelAuditor
from finshell.null_tests import NullTestConfig
from finshell.plotting import PlotConfig
from finshell.study_plotting import render_label_audit
from finshell.triple_barrier import TripleBarrierComparator, TripleBarrierConfig


class StageOrderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StageReport:
    stage: str
    passed: bool
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    figure: Path | None = None

    def __repr__(self) -> str:
        return (
            f"StageReport(stage={self.stage!r}, passed={self.passed!r}, "
            f"summary={self.summary!r}, figure={str(self.figure)!r})"
        )


class ValidationStudy:
    def __init__(
        self,
        source: pd.DataFrame | str | Path,
        *,
        roles: ColumnRoleMap,
        artifact_dir: Path | str = Path("outputs/finshell_study"),
        plots: PlotConfig | None = None,
    ) -> None:
        self.context = PipelineContext(artifact_dir=artifact_dir)
        self.plots = plots or PlotConfig(enabled=True)
        DataIngestor(DataIngestConfig(source=source, roles=roles)).run(self.context)
        self._completed: set[str] = set()
        self._barrier_config: TripleBarrierConfig | None = None

    def audit_label(
        self,
        barrier: TripleBarrierConfig | None = None,
        *,
        favorable_label: int = 1,
        label_audit: LabelAuditConfig | None = None,
        holdout: HoldoutConfig | None = None,
        null_tests: NullTestConfig | None = None,
    ) -> StageReport:
        barrier_config = barrier or TripleBarrierConfig()
        source_config = replace(
            barrier_config,
            data_key="data",
            output_key="study_triple_barrier_result",
        )
        TripleBarrierComparator(source_config).run(self.context)
        barrier_result = self.context.state["study_triple_barrier_result"]
        frame = self.context.state["data"].copy()
        frame["finshell_label"] = barrier_result["barrier_label"].to_numpy()
        frame["finshell_outcome"] = barrier_result["barrier_return"].to_numpy()
        roles = self.context.state["roles"]
        study_roles = replace(roles, label="finshell_label", outcome="finshell_outcome")
        self.context.state["data"] = frame
        self.context.state["roles"] = study_roles

        HoldoutSplitter(holdout or HoldoutConfig()).run(self.context)
        audit_config = label_audit or LabelAuditConfig()
        audit_result = LabelAuditor(replace(audit_config, data_key="development_data")).run(self.context)
        development = self.context.state["development_data"]
        label_values = pd.to_numeric(development[study_roles.label], errors="coerce")
        outcomes = pd.to_numeric(development[study_roles.outcome], errors="coerce").to_numpy(dtype=float)
        selected = label_values.eq(int(favorable_label)).to_numpy()
        null_config = null_tests or NullTestConfig()
        diagnostics = _same_count_null(
            outcomes,
            selected,
            simulations=int(null_config.random_simulations),
            stored_paths=int(null_config.stored_random_paths),
            random_seed=int(null_config.random_seed),
        )
        class_counts = dict(audit_result.summary["label_counts"])
        diagnostics["class_counts"] = class_counts
        self.context.state["label_study_diagnostics"] = diagnostics

        fail_reasons = list(audit_result.summary.get("fail_reasons", []))
        if diagnostics["favorable_count"] <= 0:
            fail_reasons.append("no_favorable_label_rows")
        if diagnostics["real_total"] <= diagnostics["random_final_p95"]:
            fail_reasons.append("label_path_not_above_random_p95")
        if diagnostics["percentile"] < float(null_config.min_random_percentile):
            fail_reasons.append("random_percentile_below_threshold")
        if diagnostics["p_value"] > float(null_config.max_p_value):
            fail_reasons.append("p_value_above_threshold")

        figure = self.context.artifact_path("plots/01_label_audit.png")
        if self.plots.enabled:
            render_label_audit(
                figure,
                real_equity=diagnostics["real_equity"],
                random_paths=diagnostics["random_equity_paths"],
                pointwise_p95=diagnostics["pointwise_p95"],
                class_counts=class_counts,
                dpi=int(self.plots.dpi),
            )
        summary = {
            "development_rows": int(len(development)),
            "quarantine_rows": int(len(self.context.state["quarantine_data"])),
            "favorable_label": int(favorable_label),
            "favorable_count": int(diagnostics["favorable_count"]),
            "real_total": float(diagnostics["real_total"]),
            "random_final_p95": float(diagnostics["random_final_p95"]),
            "percentile": float(diagnostics["percentile"]),
            "p_value": float(diagnostics["p_value"]),
            "class_counts": class_counts,
            "fail_reasons": sorted(set(fail_reasons)),
        }
        self._barrier_config = barrier_config
        self._completed.add("label_audit")
        return StageReport(
            stage="label_audit",
            passed=not summary["fail_reasons"],
            summary=summary,
            artifacts={"figure": figure} if self.plots.enabled else {},
            figure=figure if self.plots.enabled else None,
        )


def _same_count_null(
    outcomes: np.ndarray,
    selected: np.ndarray,
    *,
    simulations: int,
    stored_paths: int,
    random_seed: int,
) -> dict[str, Any]:
    values = np.asarray(outcomes, dtype=float)
    selected_mask = np.asarray(selected, dtype=bool)
    valid_positions = np.flatnonzero(np.isfinite(values))
    selected_positions = np.flatnonzero(selected_mask & np.isfinite(values))
    selected_count = int(len(selected_positions))
    real_increments = np.zeros(len(values), dtype=float)
    real_increments[selected_positions] = values[selected_positions]
    real_equity = np.cumsum(real_increments)

    rng = np.random.default_rng(random_seed)
    all_paths = np.empty((simulations, len(values)), dtype=float)
    random_totals = np.empty(simulations, dtype=float)
    stored: list[list[float]] = []
    for simulation in range(simulations):
        sampled = rng.choice(valid_positions, size=selected_count, replace=False)
        increments = np.zeros(len(values), dtype=float)
        increments[sampled] = values[sampled]
        path = np.cumsum(increments)
        all_paths[simulation] = path
        random_totals[simulation] = path[-1] if len(path) else 0.0
        if simulation < stored_paths:
            stored.append([float(value) for value in np.round(path, 12).tolist()])
    real_total = float(real_equity[-1]) if len(real_equity) else 0.0
    return {
        "favorable_count": selected_count,
        "real_equity": [float(value) for value in np.round(real_equity, 12).tolist()],
        "random_equity_paths": stored,
        "pointwise_p95": [float(value) for value in np.quantile(all_paths, 0.95, axis=0).tolist()],
        "real_total": real_total,
        "random_final_p95": float(np.quantile(random_totals, 0.95)),
        "percentile": float(np.mean(random_totals <= real_total)),
        "p_value": float(np.mean(random_totals >= real_total)),
        "random_seed": random_seed,
        "random_simulations": simulations,
    }
