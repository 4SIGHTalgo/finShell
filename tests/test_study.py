from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import finshell as fs
import finshell.study_plotting as study_plotting


def synthetic_frame(rows: int = 240) -> pd.DataFrame:
    signal = np.sin(np.linspace(0.0, 16.0 * np.pi, rows))
    close = np.empty(rows, dtype=float)
    close[0] = 100.0
    for index in range(rows - 1):
        close[index + 1] = close[index] * (1.0 + 0.012 * signal[index])
    return pd.DataFrame(
        {
            "event_time": pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC"),
            "signal_feature": signal,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "side": np.ones(rows, dtype=int),
        }
    )


def study_roles() -> fs.ColumnRoleMap:
    return fs.ColumnRoleMap(
        timestamp="event_time",
        high="high",
        low="low",
        close="close",
        side="side",
    )


def test_label_audit_generates_target_null_envelope_and_class_balance(tmp_path: Path) -> None:
    assert hasattr(fs, "ValidationStudy")
    study = fs.ValidationStudy(synthetic_frame(), roles=study_roles(), artifact_dir=tmp_path)

    report = study.audit_label(
        fs.TripleBarrierConfig(profit_take=0.01, stop_loss=0.01, vertical_bars=1),
        null_tests=fs.NullTestConfig(random_simulations=300, stored_random_paths=40, random_seed=7),
    )

    assert report.stage == "label_audit"
    assert report.passed is True
    assert report.summary["real_total"] > report.summary["random_final_p95"]
    assert report.summary["p_value"] <= 0.05
    assert report.summary["favorable_count"] >= 30
    assert set(report.summary["class_counts"]) == {"-1", "0", "1"}
    assert study.context.state["roles"].label == "finshell_label"
    assert study.context.state["roles"].outcome == "finshell_outcome"
    assert study.context.state["sealed_partitions"] == ["quarantine_holdout"]
    diagnostics = study.context.state["label_study_diagnostics"]
    assert len(diagnostics["real_equity"]) == len(study.context.state["development_data"])
    assert len(diagnostics["pointwise_p95"]) == len(study.context.state["development_data"])
    assert Path(report.figure).name == "01_label_audit.png"
    assert Path(report.figure).stat().st_size > 0


def _audited_study(tmp_path: Path) -> fs.ValidationStudy:
    study = fs.ValidationStudy(synthetic_frame(), roles=study_roles(), artifact_dir=tmp_path)
    study.audit_label(
        fs.TripleBarrierConfig(profit_take=0.01, stop_loss=0.01, vertical_bars=1),
        null_tests=fs.NullTestConfig(random_simulations=200, stored_random_paths=20, random_seed=7),
    )
    return study


def test_selector_training_requires_label_audit(tmp_path: Path) -> None:
    study = fs.ValidationStudy(synthetic_frame(), roles=study_roles(), artifact_dir=tmp_path)

    with pytest.raises(fs.StageOrderError, match="label_audit"):
        study.fit_selector(fs.LogisticSelector(features=["signal_feature"]))


def test_selector_rejects_label_and_outcome_features(tmp_path: Path) -> None:
    study = _audited_study(tmp_path)

    with pytest.raises(ValueError, match="role or target columns"):
        study.fit_selector(fs.LogisticSelector(features=["finshell_outcome"]))


def test_selector_fits_inside_cpcv_bootstrap_train_rows_only(tmp_path: Path) -> None:
    study = _audited_study(tmp_path)

    report = study.fit_selector(
        fs.LogisticSelector(features=["signal_feature"], threshold=0.60, random_state=11),
        cpcv=fs.CPCVConfig(n_groups=6, holdout_groups=2, validate_groups=1, max_splits=4),
        bootstrap=fs.FoldBlockBootstrapConfig(replicates=4, block_bars=8, random_seed=11),
    )

    assert report.stage == "selector_validation"
    assert report.passed is True
    assert report.summary["valid_bootstrap_fits"] == 16
    assert report.summary["mean_validate_total_return"] > 0.0
    assert report.summary["mean_test_total_return"] > 0.0
    for audit in study.context.state["selector_fit_audit"]:
        assert set(audit["fit_indices"]).isdisjoint(audit["validate_indices"])
        assert set(audit["fit_indices"]).isdisjoint(audit["test_indices"])
    assert study.context.state["quarantine_scored"] is False
    assert study.context.state["selector_final_fit_rows"] == list(
        range(len(study.context.state["development_data"]))
    )
    metrics = study.context.state["selector_cv_metrics"]
    assert set(metrics["partition"]) == {"validate", "test"}
    assert Path(report.figure).name == "02_cpcv_selector.png"
    assert Path(report.figure).stat().st_size > 0


def test_probability_metric_plot_range_stays_bounded() -> None:
    axis_limits = getattr(study_plotting, "metric_axis_limits", None)
    assert axis_limits is not None
    assert axis_limits("average_precision") == (0.0, 1.05)
    assert axis_limits("total_return") is None


def _trained_study(tmp_path: Path) -> fs.ValidationStudy:
    study = _audited_study(tmp_path)
    study.fit_selector(
        fs.LogisticSelector(features=["signal_feature"], threshold=0.25, random_state=13),
        cpcv=fs.CPCVConfig(n_groups=6, holdout_groups=2, validate_groups=1, max_splits=4),
        bootstrap=fs.FoldBlockBootstrapConfig(replicates=4, block_bars=8, random_seed=13),
    )
    return study


def test_oos_audit_requires_selector_validation(tmp_path: Path) -> None:
    study = _audited_study(tmp_path)

    with pytest.raises(fs.StageOrderError, match="selector_validation"):
        study.audit_oos()


def test_oos_selected_path_beats_random_p95_and_no_selector(tmp_path: Path) -> None:
    study = _trained_study(tmp_path)
    coefficients_before = study.context.state["selector_model"].coef_.copy()

    report = study.audit_oos(
        null_tests=fs.NullTestConfig(random_simulations=300, stored_random_paths=40, random_seed=17)
    )

    assert report.stage == "oos_audit"
    assert report.passed is True
    assert report.summary["selected_total"] > report.summary["random_final_p95"]
    assert report.summary["selected_total"] > report.summary["no_selector_total"]
    assert report.summary["p_value"] <= 0.05
    assert report.summary["selected_count"] > 0
    assert np.array_equal(coefficients_before, study.context.state["selector_model"].coef_)
    assert study.context.state["quarantine_scored"] is True
    diagnostics = study.context.state["oos_study_diagnostics"]
    quarantine_rows = len(study.context.state["quarantine_data"])
    assert len(diagnostics["selected_equity"]) == quarantine_rows
    assert len(diagnostics["pointwise_p95"]) == quarantine_rows
    assert len(diagnostics["no_selector_equity"]) == quarantine_rows
    assert Path(report.figure).name == "03_oos_audit.png"
    assert Path(report.figure).stat().st_size > 0


def _oos_study(tmp_path: Path) -> fs.ValidationStudy:
    study = _trained_study(tmp_path)
    study.audit_oos(
        null_tests=fs.NullTestConfig(random_simulations=200, stored_random_paths=20, random_seed=19)
    )
    return study


def test_economic_validation_requires_oos_audit(tmp_path: Path) -> None:
    study = _trained_study(tmp_path)

    with pytest.raises(fs.StageOrderError, match="oos_audit"):
        study.validate_economics()


def test_economic_validation_bootstraps_selected_oos_and_barrier_geometry(tmp_path: Path) -> None:
    study = _oos_study(tmp_path)

    report = study.validate_economics(paths=200, block_bars=4, random_seed=23)

    assert report.stage == "economic_validation"
    assert report.passed is True
    assert report.summary["source_trade_count"] == study.context.state["oos_study_diagnostics"][
        "favorable_count"
    ]
    assert report.summary["terminal_p05"] <= report.summary["terminal_p50"]
    assert report.summary["terminal_p50"] <= report.summary["terminal_p95"]
    assert report.summary["terminal_p50"] > 0.0
    bootstrap_paths = np.asarray(
        study.context.state["economic_study_diagnostics"]["bootstrap_paths"]
    )
    assert np.ptp(bootstrap_paths[:, -1]) > 0.0
    geometry = study.context.state["economic_study_diagnostics"]["barrier_geometry"]
    assert geometry["upper_barrier"] == pytest.approx(
        geometry["entry_price"] * (1.0 + 0.01)
    )
    assert geometry["lower_barrier"] == pytest.approx(
        geometry["entry_price"] * (1.0 - 0.01)
    )
    assert geometry["vertical_x"] == 1
    assert Path(report.figure).name == "04_economic_monte_carlo.png"
    assert Path(report.figure).stat().st_size > 0
