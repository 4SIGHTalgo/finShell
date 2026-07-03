from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from finshell.core import PipelineContext
from finshell.bootstrap import FoldBlockBootstrap, FoldBlockBootstrapConfig
from finshell.cpcv import CPCVConfig, CPCVPurgeEmbargo
from finshell.holdout import HoldoutConfig, HoldoutSplitter
from finshell.ingestion import ColumnRoleMap, DataIngestConfig, DataIngestor
from finshell.label_audit import LabelAuditConfig, LabelAuditor
from finshell.null_tests import NullTestConfig
from finshell.plotting import PlotConfig
from finshell.study_plotting import render_label_audit, render_oos_audit, render_selector_cv
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


@dataclass(frozen=True, slots=True)
class LogisticSelector:
    features: Sequence[str]
    threshold: float = 0.50
    c: float = 1.0
    max_iter: int = 500
    random_state: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(str(value) for value in self.features))
        if not self.features:
            raise ValueError("features must not be empty")
        if not 0.0 < float(self.threshold) < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.c <= 0.0:
            raise ValueError("c must be > 0")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")

    def build_estimator(self) -> Any:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=float(self.c),
            max_iter=int(self.max_iter),
            random_state=int(self.random_state),
            solver="lbfgs",
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

    def fit_selector(
        self,
        selector: LogisticSelector,
        *,
        cpcv: CPCVConfig | None = None,
        bootstrap: FoldBlockBootstrapConfig | None = None,
        favorable_label: int = 1,
    ) -> StageReport:
        self._require_stage("label_audit")
        development = self.context.state["development_data"]
        roles = self.context.state["roles"]
        forbidden = set(roles.named_columns().values())
        invalid_features = sorted(set(selector.features).intersection(forbidden))
        if invalid_features:
            raise ValueError("selector features include role or target columns: " + ", ".join(invalid_features))
        missing = sorted(set(selector.features).difference(development.columns))
        if missing:
            raise ValueError("missing selector features: " + ", ".join(missing))

        CPCVPurgeEmbargo(cpcv or CPCVConfig()).run(self.context)
        FoldBlockBootstrap(bootstrap or FoldBlockBootstrapConfig()).run(self.context)
        folds = self.context.state["cpcv_folds"]
        plans = {plan.fold_name: plan for plan in self.context.state["bootstrap_plans"]}
        target = pd.to_numeric(development[roles.label], errors="coerce").eq(int(favorable_label)).astype(int)
        outcomes = pd.to_numeric(development[roles.outcome], errors="coerce")
        metrics_rows: list[dict[str, Any]] = []
        fit_audit: list[dict[str, Any]] = []
        skipped_fits = 0

        for fold in folds:
            plan = plans[fold.name]
            for path in plan.paths:
                fit_indices = [int(value) for value in path.indices]
                x_train = development.iloc[fit_indices][list(selector.features)].to_numpy(dtype=float)
                y_train = target.iloc[fit_indices].to_numpy(dtype=int)
                finite_train = np.isfinite(x_train).all(axis=1)
                x_train = x_train[finite_train]
                y_train = y_train[finite_train]
                if len(np.unique(y_train)) < 2:
                    skipped_fits += 1
                    continue
                estimator = selector.build_estimator()
                estimator.fit(x_train, y_train)
                fit_audit.append(
                    {
                        "fold": fold.name,
                        "replicate": int(path.replicate),
                        "fit_indices": sorted(set(fit_indices)),
                        "validate_indices": list(fold.validate_indices),
                        "test_indices": list(fold.test_indices),
                    }
                )
                for partition, indices in (
                    ("validate", fold.validate_indices),
                    ("test", fold.test_indices),
                ):
                    x_eval = development.iloc[indices][list(selector.features)].to_numpy(dtype=float)
                    y_eval = target.iloc[indices].to_numpy(dtype=int)
                    eval_outcomes = outcomes.iloc[indices].to_numpy(dtype=float)
                    finite_eval = np.isfinite(x_eval).all(axis=1) & np.isfinite(eval_outcomes)
                    x_eval = x_eval[finite_eval]
                    y_eval = y_eval[finite_eval]
                    eval_outcomes = eval_outcomes[finite_eval]
                    scores = _positive_scores(estimator, x_eval)
                    selected = scores >= float(selector.threshold)
                    selected_values = eval_outcomes[selected]
                    metrics_rows.append(
                        {
                            "fold": fold.name,
                            "replicate": int(path.replicate),
                            "partition": partition,
                            "rows": int(len(y_eval)),
                            "prevalence": float(np.mean(y_eval)) if len(y_eval) else float("nan"),
                            "average_precision": _average_precision(y_eval, scores),
                            "roc_auc": _roc_auc(y_eval, scores),
                            "selected_count": int(selected.sum()),
                            "total_return": float(np.sum(selected_values)),
                            "mean_return": float(np.mean(selected_values)) if len(selected_values) else float("nan"),
                        }
                    )

        metrics = pd.DataFrame(metrics_rows)
        valid_fit_count = len(fit_audit)
        fail_reasons: list[str] = []
        if valid_fit_count <= 0:
            fail_reasons.append("no_valid_bootstrap_fits")
        validate_mean = _partition_metric_mean(metrics, "validate", "total_return")
        test_mean = _partition_metric_mean(metrics, "test", "total_return")
        if not np.isfinite(validate_mean) or validate_mean <= 0.0:
            fail_reasons.append("non_positive_validate_economics")
        if not np.isfinite(test_mean) or test_mean <= 0.0:
            fail_reasons.append("non_positive_test_economics")

        final_rows = np.flatnonzero(
            np.isfinite(development[list(selector.features)].to_numpy(dtype=float)).all(axis=1)
            & target.notna().to_numpy()
        )
        final_estimator = selector.build_estimator()
        final_estimator.fit(
            development.iloc[final_rows][list(selector.features)].to_numpy(dtype=float),
            target.iloc[final_rows].to_numpy(dtype=int),
        )
        self.context.state["selector_model"] = final_estimator
        self.context.state["selector_spec"] = selector
        self.context.state["selector_cv_metrics"] = metrics
        self.context.state["selector_fit_audit"] = fit_audit
        self.context.state["selector_final_fit_rows"] = [int(value) for value in final_rows.tolist()]
        self.context.state["quarantine_scored"] = False

        figure = self.context.artifact_path("plots/02_cpcv_selector.png")
        if self.plots.enabled:
            render_selector_cv(figure, metrics=metrics, dpi=int(self.plots.dpi))
        summary = {
            "fold_count": int(len(folds)),
            "bootstrap_replicates_per_fold": int(len(next(iter(plans.values())).paths)) if plans else 0,
            "valid_bootstrap_fits": int(valid_fit_count),
            "skipped_bootstrap_fits": int(skipped_fits),
            "mean_validate_total_return": validate_mean,
            "mean_test_total_return": test_mean,
            "mean_validate_average_precision": _partition_metric_mean(metrics, "validate", "average_precision"),
            "mean_test_average_precision": _partition_metric_mean(metrics, "test", "average_precision"),
            "threshold": float(selector.threshold),
            "features": list(selector.features),
            "fail_reasons": fail_reasons,
        }
        self._completed.add("selector_validation")
        return StageReport(
            stage="selector_validation",
            passed=not fail_reasons,
            summary=summary,
            artifacts={"figure": figure} if self.plots.enabled else {},
            figure=figure if self.plots.enabled else None,
        )

    def audit_oos(self, *, null_tests: NullTestConfig | None = None) -> StageReport:
        self._require_stage("selector_validation")
        quarantine = self.context.state["quarantine_data"].copy()
        roles = self.context.state["roles"]
        selector = self.context.state["selector_spec"]
        estimator = self.context.state["selector_model"]
        features = quarantine[list(selector.features)].to_numpy(dtype=float)
        finite_features = np.isfinite(features).all(axis=1)
        scores = np.full(len(quarantine), np.nan, dtype=float)
        scores[finite_features] = _positive_scores(estimator, features[finite_features])
        selected = np.isfinite(scores) & (scores >= float(selector.threshold))
        outcomes = pd.to_numeric(quarantine[roles.outcome], errors="coerce").to_numpy(dtype=float)
        null_config = null_tests or NullTestConfig()
        diagnostics = _same_count_null(
            outcomes,
            selected,
            simulations=int(null_config.random_simulations),
            stored_paths=int(null_config.stored_random_paths),
            random_seed=int(null_config.random_seed),
        )
        no_selector_increments = np.where(np.isfinite(outcomes), outcomes, 0.0)
        no_selector_equity = np.cumsum(no_selector_increments)
        no_selector_total = float(no_selector_equity[-1]) if len(no_selector_equity) else 0.0
        diagnostics["selected_equity"] = diagnostics.pop("real_equity")
        diagnostics["no_selector_equity"] = [
            float(value) for value in np.round(no_selector_equity, 12).tolist()
        ]
        self.context.state["oos_study_diagnostics"] = diagnostics
        scored = quarantine.copy()
        scored["model_score"] = scores
        scored["selected"] = selected
        self.context.state["quarantine_scored_data"] = scored
        self.context.state["quarantine_scored"] = True

        fail_reasons: list[str] = []
        if diagnostics["favorable_count"] <= 0:
            fail_reasons.append("no_oos_selected_rows")
        if diagnostics["real_total"] <= diagnostics["random_final_p95"]:
            fail_reasons.append("selected_path_not_above_random_p95")
        if diagnostics["real_total"] <= no_selector_total:
            fail_reasons.append("selected_path_not_above_no_selector")
        if diagnostics["percentile"] < float(null_config.min_random_percentile):
            fail_reasons.append("random_percentile_below_threshold")
        if diagnostics["p_value"] > float(null_config.max_p_value):
            fail_reasons.append("p_value_above_threshold")

        figure = self.context.artifact_path("plots/03_oos_audit.png")
        if self.plots.enabled:
            render_oos_audit(
                figure,
                selected_equity=diagnostics["selected_equity"],
                random_paths=diagnostics["random_equity_paths"],
                pointwise_p95=diagnostics["pointwise_p95"],
                no_selector_equity=diagnostics["no_selector_equity"],
                dpi=int(self.plots.dpi),
            )
        selected_values = outcomes[selected & np.isfinite(outcomes)]
        summary = {
            "quarantine_rows": int(len(quarantine)),
            "selected_count": int(diagnostics["favorable_count"]),
            "selected_total": float(diagnostics["real_total"]),
            "selected_mean": float(np.mean(selected_values)) if len(selected_values) else float("nan"),
            "no_selector_total": no_selector_total,
            "random_final_p95": float(diagnostics["random_final_p95"]),
            "percentile": float(diagnostics["percentile"]),
            "p_value": float(diagnostics["p_value"]),
            "fail_reasons": fail_reasons,
        }
        self._completed.add("oos_audit")
        return StageReport(
            stage="oos_audit",
            passed=not fail_reasons,
            summary=summary,
            artifacts={"figure": figure} if self.plots.enabled else {},
            figure=figure if self.plots.enabled else None,
        )

    def _require_stage(self, stage: str) -> None:
        if stage not in self._completed:
            raise StageOrderError(f"stage {stage!r} must complete first")


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


def _positive_scores(estimator: Any, features: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    positive_index = int(np.flatnonzero(estimator.classes_ == 1)[0])
    return probabilities[:, positive_index].astype(float)


def _average_precision(target: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if not len(target) or int(np.sum(target)) <= 0:
        return float("nan")
    return float(average_precision_score(target, scores))


def _roc_auc(target: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(target)) < 2:
        return float("nan")
    return float(roc_auc_score(target, scores))


def _partition_metric_mean(metrics: pd.DataFrame, partition: str, column: str) -> float:
    if metrics.empty:
        return float("nan")
    values = pd.to_numeric(
        metrics.loc[metrics["partition"].eq(partition), column],
        errors="coerce",
    ).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")
