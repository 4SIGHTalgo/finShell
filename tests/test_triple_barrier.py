from __future__ import annotations

from pathlib import Path

import pandas as pd

from finshell.core import PipelineContext
from finshell.ingestion import ColumnRoleMap
from finshell.triple_barrier import TripleBarrierComparator, TripleBarrierConfig


def _context(tmp_path: Path, high: list[float], low: list[float], close: list[float], side: list[int] | None = None) -> PipelineContext:
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=len(close), freq="1h", tz="UTC"),
            "high": high,
            "low": low,
            "close": close,
            "tb_label": [0] * len(close),
        }
    )
    side_col = None
    if side is not None:
        frame["side"] = side
        side_col = "side"
    context = PipelineContext(tmp_path)
    context.state["development_data"] = frame
    context.state["roles"] = ColumnRoleMap(
        timestamp="event_time",
        label="tb_label",
        high="high",
        low="low",
        close="close",
        side=side_col,
    )
    return context


def test_triple_barrier_labels_favorable_upper_hit_for_long_side(tmp_path: Path) -> None:
    context = _context(tmp_path, high=[100, 103, 101], low=[99, 100, 100], close=[100, 101, 101])

    result = TripleBarrierComparator(TripleBarrierConfig(profit_take=0.02, stop_loss=0.02, vertical_bars=2)).run(context)

    assert result.passed is True
    assert context.state["triple_barrier_result"]["barrier_label"].iloc[0] == 1
    assert context.state["triple_barrier_result"]["exit_reason"].iloc[0] == "take_profit"


def test_triple_barrier_labels_adverse_lower_hit_for_long_side(tmp_path: Path) -> None:
    context = _context(tmp_path, high=[100, 101, 101], low=[99, 97, 100], close=[100, 100, 100])

    TripleBarrierComparator(TripleBarrierConfig(profit_take=0.02, stop_loss=0.02, vertical_bars=2)).run(context)

    assert context.state["triple_barrier_result"]["barrier_label"].iloc[0] == -1
    assert context.state["triple_barrier_result"]["exit_reason"].iloc[0] == "stop_loss"


def test_triple_barrier_labels_vertical_timeout(tmp_path: Path) -> None:
    context = _context(tmp_path, high=[100, 101, 101], low=[99, 99, 99], close=[100, 100, 100])

    TripleBarrierComparator(TripleBarrierConfig(profit_take=0.05, stop_loss=0.05, vertical_bars=2)).run(context)

    assert context.state["triple_barrier_result"]["barrier_label"].iloc[0] == 0
    assert context.state["triple_barrier_result"]["exit_reason"].iloc[0] == "timeout"


def test_triple_barrier_is_side_aware_for_short_side(tmp_path: Path) -> None:
    context = _context(
        tmp_path,
        high=[100, 101, 101],
        low=[99, 97, 100],
        close=[100, 100, 100],
        side=[-1, -1, -1],
    )

    TripleBarrierComparator(TripleBarrierConfig(profit_take=0.02, stop_loss=0.02, vertical_bars=2)).run(context)

    assert context.state["triple_barrier_result"]["barrier_label"].iloc[0] == 1
    assert context.state["triple_barrier_result"]["exit_reason"].iloc[0] == "take_profit"


def test_triple_barrier_can_be_omitted(tmp_path: Path) -> None:
    context = _context(tmp_path, high=[100, 101], low=[99, 99], close=[100, 100])

    result = TripleBarrierComparator(TripleBarrierConfig(enabled=False)).run(context)

    assert result.passed is True
    assert result.summary["skipped"] is True
