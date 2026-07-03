from __future__ import annotations

from pathlib import Path

import pandas as pd

from finshell.core import PipelineContext
from finshell.holdout import HoldoutConfig, HoldoutSplitter
from finshell.ingestion import ColumnRoleMap


def _context(tmp_path: Path, rows: int = 10) -> PipelineContext:
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC"),
            "tb_label": [1 if idx % 2 else -1 for idx in range(rows)],
        }
    )
    context = PipelineContext(tmp_path)
    context.state["data"] = frame
    context.state["roles"] = ColumnRoleMap(timestamp="event_time", label="tb_label")
    return context


def test_holdout_splitter_uses_last_20_percent_by_time_by_default(tmp_path: Path) -> None:
    context = _context(tmp_path, rows=10)

    result = HoldoutSplitter().run(context)

    assert result.passed is True
    assert len(context.state["development_data"]) == 8
    assert len(context.state["quarantine_data"]) == 2
    assert context.state["quarantine_data"]["event_time"].min() == pd.Timestamp("2026-01-01T08:00:00Z")
    assert result.summary["quarantine_fraction"] == 0.2


def test_holdout_splitter_supports_explicit_quarantine_dates(tmp_path: Path) -> None:
    context = _context(tmp_path, rows=6)

    result = HoldoutSplitter(
        HoldoutConfig(
            quarantine_start="2026-01-01T03:00:00Z",
            quarantine_end_exclusive="2026-01-01T05:00:00Z",
        )
    ).run(context)

    assert result.passed is True
    assert context.state["quarantine_data"]["event_time"].tolist() == [
        pd.Timestamp("2026-01-01T03:00:00Z"),
        pd.Timestamp("2026-01-01T04:00:00Z"),
    ]
    assert result.summary["mode"] == "explicit_window"


def test_holdout_splitter_records_sealed_partitions_and_excludes_holdout_from_development(tmp_path: Path) -> None:
    context = _context(tmp_path, rows=5)

    HoldoutSplitter(HoldoutConfig(quarantine_fraction=0.4)).run(context)

    dev_times = set(context.state["development_data"]["event_time"])
    holdout_times = set(context.state["quarantine_data"]["event_time"])
    assert dev_times.isdisjoint(holdout_times)
    assert context.state["sealed_partitions"] == ["quarantine_holdout"]
