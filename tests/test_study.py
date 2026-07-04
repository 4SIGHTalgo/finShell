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


def _audited_study(tmp_path: Path, *, vertical_bars: int = 1) -> fs.ValidationStudy:
    study = fs.ValidationStudy(synthetic_frame(), roles=study_roles(), artifact_dir=tmp_path)
    study.audit_label(
        fs.TripleBarrierConfig(
            profit_take=0.01,
            stop_loss=0.01,
            vertical_bars=vertical_bars,
        ),
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
    nested = study.context.state["selector_cv_nested_distributions"]
    assert len(nested["folds"]) == 4
    assert len(nested["outer"]["validate"]) == 4
    assert len(nested["outer"]["test"]) == 4
    for fold in nested["folds"]:
        assert fold["validate_group_indices"]
        assert fold["test_group_indices"]
        assert len(fold["validate"]) == 4
        assert len(fold["test"]) == 4
        assert fold["validate_mean"] == pytest.approx(np.mean(fold["validate"]))
        assert fold["test_mean"] == pytest.approx(np.mean(fold["test"]))
    assert Path(report.figure).name == "02_cpcv_selector.png"
    assert Path(report.figure).stat().st_size > 0


def test_probability_metric_plot_range_stays_bounded() -> None:
    axis_limits = getattr(study_plotting, "metric_axis_limits", None)
    assert axis_limits is not None
    assert axis_limits("average_precision") == (0.0, 1.05)
    assert axis_limits("total_return") is None


def _trained_study(tmp_path: Path, *, vertical_bars: int = 1) -> fs.ValidationStudy:
    study = _audited_study(tmp_path, vertical_bars=vertical_bars)
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
    quarantine_outcomes = pd.to_numeric(
        study.context.state["quarantine_data"][study.context.state["roles"].outcome],
        errors="coerce",
    ).fillna(0.0)
    np.testing.assert_allclose(
        diagnostics["no_selector_equity"],
        quarantine_outcomes.cumsum().to_numpy(dtype=float),
    )
    assert Path(report.figure).name == "03_oos_audit.png"
    assert Path(report.figure).stat().st_size > 0


def _oos_study(tmp_path: Path) -> fs.ValidationStudy:
    study = _trained_study(tmp_path, vertical_bars=8)
    study.audit_oos(
        null_tests=fs.NullTestConfig(random_simulations=200, stored_random_paths=20, random_seed=19)
    )
    return study


def test_economic_validation_requires_oos_audit(tmp_path: Path) -> None:
    study = _trained_study(tmp_path)

    with pytest.raises(fs.StageOrderError, match="oos_audit"):
        study.validate_economics()


def test_economic_validation_bootstraps_oos_account_paths_inside_barriers(tmp_path: Path) -> None:
    study = _oos_study(tmp_path)

    report = study.validate_economics(
        paths=200,
        block_bars=4,
        random_seed=23,
        initial_balance=100_000.0,
        upper_balance=102_000.0,
        lower_balance=98_000.0,
        max_trades=24,
    )

    assert report.stage == "economic_validation"
    assert report.passed is (report.summary["terminal_p50"] > 0.0)
    assert report.summary["source_trade_count"] == study.context.state["oos_study_diagnostics"][
        "favorable_count"
    ]
    assert 0.0 <= report.summary["upper_hit_probability"] <= 1.0
    assert 1.0 <= report.summary["median_resolution_trades"] <= 24.0
    diagnostics = study.context.state["economic_study_diagnostics"]
    selected = study.context.state["quarantine_scored_data"]["selected"].astype(bool)
    roles = study.context.state["roles"]
    expected_source = pd.to_numeric(
        study.context.state["quarantine_scored_data"].loc[selected, roles.outcome],
        errors="coerce",
    ).dropna()
    assert diagnostics["source_returns"] == pytest.approx(expected_source.tolist())
    paths = np.asarray(diagnostics["account_balance_paths"], dtype=float)
    states = diagnostics["resolution_states"]
    resolution_trades = diagnostics["resolution_trades"]
    assert paths.shape == (200, 25)
    assert len(states) == len(resolution_trades) == 200
    assert set(states) <= {"upper", "lower", "vertical"}
    assert sum(diagnostics["state_counts"].values()) == 200
    assert report.summary["upper_hit_probability"] == pytest.approx(
        states.count("upper") / 200.0
    )
    assert report.summary["median_resolution_trades"] == pytest.approx(
        np.median(resolution_trades)
    )
    assert diagnostics["initial_balance"] == 100_000.0
    upper = diagnostics["upper_balance"]
    lower = diagnostics["lower_balance"]
    for path, state, resolved_at in zip(paths, states, resolution_trades):
        assert path[0] == pytest.approx(100_000.0)
        terminal = path[int(resolved_at)]
        if state == "upper":
            assert terminal == pytest.approx(upper)
        elif state == "lower":
            assert terminal == pytest.approx(lower)
        else:
            assert resolved_at == 24
    assert Path(report.figure).name == "04_economic_monte_carlo.png"
    assert Path(report.figure).stat().st_size > 0
