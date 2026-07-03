from __future__ import annotations

from pathlib import Path

import pandas as pd

from finshell import ColumnRoleMap, FullPipeline
from finshell.bootstrap import FoldBlockBootstrapConfig
from finshell.cpcv import CPCVConfig
from finshell.label_audit import LabelAuditConfig
from finshell.null_tests import NullTestConfig
from finshell.risk_metrics import RiskMetricsConfig


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
