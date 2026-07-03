from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from finshell.core import PipelineContext
from finshell.ingestion import ColumnRoleMap
from finshell.risk_metrics import RiskMetrics, RiskMetricsConfig


def _context(tmp_path: Path) -> PipelineContext:
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC"),
            "tb_label": [1, -1, 1, 0, 1],
            "selected": [True, True, True, False, False],
            "net_return": [0.10, -0.05, 0.02, -0.20, 0.30],
        }
    )
    context = PipelineContext(tmp_path)
    context.state["development_data"] = frame
    context.state["roles"] = ColumnRoleMap(
        timestamp="event_time",
        label="tb_label",
        selected="selected",
        outcome="net_return",
    )
    return context


def test_risk_metrics_compute_selected_path_summary(tmp_path: Path) -> None:
    result = RiskMetrics(RiskMetricsConfig(min_selected=3)).run(_context(tmp_path))

    assert result.passed is True
    assert result.summary["selected_count"] == 3
    assert np.isclose(result.summary["mean_return"], np.mean([0.10, -0.05, 0.02]))
    assert np.isclose(result.summary["hit_rate"], 2 / 3)
    assert result.summary["max_drawdown"] == 0.05


def test_risk_metrics_compute_sortino_and_sharpe_like_values(tmp_path: Path) -> None:
    result = RiskMetrics(RiskMetricsConfig(min_selected=3, annualization_factor=1.0)).run(_context(tmp_path))

    assert np.isfinite(result.summary["volatility"])
    assert np.isfinite(result.summary["sharpe_like"])
    assert np.isfinite(result.summary["sortino_like"])


def test_risk_metrics_fails_when_selected_count_is_too_small(tmp_path: Path) -> None:
    result = RiskMetrics(RiskMetricsConfig(min_selected=4)).run(_context(tmp_path))

    assert result.passed is False
    assert "insufficient_selected_count" in result.summary["fail_reasons"]
