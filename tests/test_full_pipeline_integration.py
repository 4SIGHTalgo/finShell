from __future__ import annotations

from pathlib import Path

import pandas as pd

from finshell import ColumnRoleMap, FullPipeline
from finshell.bootstrap import FoldBlockBootstrapConfig
from finshell.cpcv import CPCVConfig
from finshell.label_audit import LabelAuditConfig
from finshell.null_tests import NullTestConfig
from finshell.plotting import PlotConfig
from finshell.risk_metrics import RiskMetricsConfig
from finshell.triple_barrier import TripleBarrierConfig


def test_validation_pipeline_runs_end_to_end_on_role_mapped_data(tmp_path: Path) -> None:
    rows = 30
    outcomes = [0.0] * rows
    selected = [False] * rows
    for index in range(20, 24):
        outcomes[index] = 0.10 + (index - 20) * 0.01
        selected[index] = True
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC"),
            "tb_label": [1 if value > 0 else 0 for value in outcomes],
            "selected": selected,
            "net_return": outcomes,
        }
    )
    roles = ColumnRoleMap(
        timestamp="event_time",
        label="tb_label",
        selected="selected",
        outcome="net_return",
    )

    pipeline = FullPipeline.validation_pipeline(
        source=frame,
        roles=roles,
        artifact_dir=tmp_path,
        label_audit=LabelAuditConfig(min_labeled_rows=20, allowed_label_values=(0, 1)),
        cpcv=CPCVConfig(n_groups=4, holdout_groups=2, validate_groups=1, max_splits=2),
        bootstrap=FoldBlockBootstrapConfig(replicates=2, block_bars=2),
        null_tests=NullTestConfig(random_simulations=200, random_seed=5),
        risk_metrics=RiskMetricsConfig(min_selected=4),
    )

    result = pipeline.run()

    assert result.passed is True
    assert result.manifest["components"][-1]["component"] == "risk_metrics"
    assert result.context.state["sealed_partitions"] == ["quarantine_holdout"]
    assert len(result.context.state["cpcv_folds"]) == 2
    assert result.context.state["null_test_summary"]["observed_total"] > 0.0
    assert result.context.state["risk_metrics_summary"]["selected_count"] == 4


def test_validation_pipeline_opt_in_plots_generate_at_each_stage(tmp_path: Path) -> None:
    rows = 40
    outcomes = [0.0] * rows
    selected = [False] * rows
    for index in range(24, 28):
        outcomes[index] = 0.10 + (index - 24) * 0.01
        selected[index] = True
    close = [100.0 + (index % 4) * 0.2 for index in range(rows)]
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC"),
            "tb_label": [1 if value > 0 else 0 for value in outcomes],
            "selected": selected,
            "net_return": outcomes,
            "high": [value * 1.03 if index % 5 == 0 else value * 1.005 for index, value in enumerate(close)],
            "low": [value * 0.97 if index % 7 == 0 else value * 0.995 for index, value in enumerate(close)],
            "close": close,
        }
    )
    roles = ColumnRoleMap(
        timestamp="event_time",
        label="tb_label",
        selected="selected",
        outcome="net_return",
        high="high",
        low="low",
        close="close",
    )

    pipeline = FullPipeline.validation_pipeline(
        source=frame,
        roles=roles,
        artifact_dir=tmp_path,
        label_audit=LabelAuditConfig(min_labeled_rows=20, allowed_label_values=(0, 1)),
        cpcv=CPCVConfig(n_groups=4, holdout_groups=2, validate_groups=1, max_splits=3),
        bootstrap=FoldBlockBootstrapConfig(replicates=5, block_bars=2),
        null_tests=NullTestConfig(random_simulations=300, random_seed=5),
        triple_barrier=TripleBarrierConfig(profit_take=0.02, stop_loss=0.02, vertical_bars=3),
        plots=PlotConfig(enabled=True, max_paths=8, bootstrap_block_bars=2),
        risk_metrics=RiskMetricsConfig(min_selected=4),
    )

    result = pipeline.run()

    assert result.passed is True
    assert [component.component for component in result.components] == [
        "data_ingestion",
        "label_audit",
        "label_diagnostics_plot",
        "holdout_split",
        "cpcv_purge_embargo",
        "fold_block_bootstrap",
        "cpcv_diagnostics_plot",
        "null_tests",
        "null_test_diagnostics_plot",
        "triple_barrier",
        "triple_barrier_diagnostics_plot",
        "risk_metrics",
    ]
    artifacts = {
        Path(path).name
        for component in result.components
        for path in component.artifacts.values()
    }
    assert artifacts == {
        "label_diagnostics.png",
        "cpcv_distributions.png",
        "null_test_equity_paths.png",
        "triple_barrier_bootstrap_paths.png",
    }
    assert all((tmp_path / "plots" / name).stat().st_size > 0 for name in artifacts)
