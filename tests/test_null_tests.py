from __future__ import annotations

from pathlib import Path

import pandas as pd

from finshell.core import PipelineContext
from finshell.ingestion import ColumnRoleMap
from finshell.null_tests import NullTestConfig, NullTestSuite


def _context(tmp_path: Path, outcomes: list[float], selected: list[bool]) -> PipelineContext:
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=len(outcomes), freq="1h", tz="UTC"),
            "tb_label": [1 if value > 0 else -1 for value in outcomes],
            "selected": selected,
            "net_return": outcomes,
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


def test_null_suite_passes_when_real_selected_path_beats_random_p95(tmp_path: Path) -> None:
    outcomes = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.10, 0.11, 0.12, 0.13]
    selected = [False, False, False, False, False, False, True, True, True, True]

    result = NullTestSuite(NullTestConfig(random_simulations=300, random_seed=3)).run(
        _context(tmp_path, outcomes, selected)
    )

    assert result.passed is True
    assert result.summary["observed_total"] > result.summary["same_count_random_p95"]
    assert result.summary["observed_random_percentile"] >= 0.95


def test_null_suite_fails_when_real_selected_path_does_not_beat_random_p95(tmp_path: Path) -> None:
    outcomes = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.10, 0.11, 0.12, 0.13]
    selected = [True, True, True, True, False, False, False, False, False, False]

    result = NullTestSuite(NullTestConfig(random_simulations=300, random_seed=3)).run(
        _context(tmp_path, outcomes, selected)
    )

    assert result.passed is False
    assert "real_path_not_above_random_p95" in result.summary["fail_reasons"]


def test_null_suite_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    outcomes = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    selected = [False, False, False, True, True, True]
    config = NullTestConfig(random_simulations=100, random_seed=17)

    result_a = NullTestSuite(config).run(_context(tmp_path / "a", outcomes, selected))
    result_b = NullTestSuite(config).run(_context(tmp_path / "b", outcomes, selected))

    assert result_a.summary["same_count_random_p95"] == result_b.summary["same_count_random_p95"]
    assert result_a.summary["random_totals"] == result_b.summary["random_totals"]


def test_null_suite_retains_exact_sampled_row_positions(tmp_path: Path) -> None:
    outcomes = [-0.02, float("nan"), 0.0, 0.01, 0.02, 0.03]
    selected = [False, False, False, True, True, True]
    context = _context(tmp_path, outcomes, selected)

    result = NullTestSuite(NullTestConfig(random_simulations=5, random_seed=17)).run(context)

    diagnostics = context.state["null_test_diagnostics"]
    frame = context.state["development_data"]
    assert diagnostics["selected_row_indices"] == [3, 4, 5]
    reconstructed = [
        float(frame.iloc[indices]["net_return"].sum())
        for indices in diagnostics["random_row_indices"]
    ]
    assert reconstructed == result.summary["random_totals"]
    assert diagnostics["random_seed"] == 17


def test_null_suite_fails_closed_without_selected_or_outcome_roles(tmp_path: Path) -> None:
    context = _context(tmp_path, [0.1, -0.1, 0.2], [True, False, True])
    context.state["roles"] = ColumnRoleMap(timestamp="event_time", label="tb_label")

    result = NullTestSuite().run(context)

    assert result.passed is False
    assert "missing_selected_or_outcome_role" in result.summary["fail_reasons"]
