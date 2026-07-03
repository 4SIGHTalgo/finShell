from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import finshell as fs


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
