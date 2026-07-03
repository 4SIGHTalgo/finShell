from __future__ import annotations

from pathlib import Path

import pandas as pd

from finshell.bootstrap import FoldBlockBootstrap, FoldBlockBootstrapConfig, infer_block_bars
from finshell.core import PipelineContext
from finshell.cpcv import CPCVFold
from finshell.ingestion import ColumnRoleMap


def _multi_source_context(tmp_path: Path) -> PipelineContext:
    timestamps = pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC")
    rows = []
    for timestamp in timestamps:
        for asset in ("BTC", "ETH"):
            rows.append({"event_time": timestamp, "asset": asset, "tb_label": 1, "net_return": 0.01})
    frame = pd.DataFrame(rows)
    context = PipelineContext(tmp_path)
    context.state["development_data"] = frame
    context.state["roles"] = ColumnRoleMap(timestamp="event_time", label="tb_label", outcome="net_return", group="asset")
    context.state["cpcv_folds"] = [
        CPCVFold(
            name="fold_1",
            validate_group_indices=[1],
            test_group_indices=[2],
            train_indices=list(range(len(frame))),
            validate_indices=[],
            test_indices=[],
            heldout_intervals=[],
        )
    ]
    return context


def test_fold_block_bootstrap_keeps_sources_synchronized_by_timestamp(tmp_path: Path) -> None:
    context = _multi_source_context(tmp_path)

    FoldBlockBootstrap(FoldBlockBootstrapConfig(replicates=2, block_bars=2, random_seed=7)).run(context)

    plan = context.state["bootstrap_plans"][0]
    sample = plan.resample(context.state["development_data"], plan.paths[0])
    for _timestamp, group in sample.groupby("event_time", sort=False):
        sampled_assets = group["asset"].tolist()
        chunks = [sampled_assets[index : index + 2] for index in range(0, len(sampled_assets), 2)]
        assert all(chunk == ["BTC", "ETH"] for chunk in chunks)


def test_infer_block_bars_respects_horizon_purge_embargo_and_floor() -> None:
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=12, freq="15min", tz="UTC"))
    label_end = timestamps + pd.Timedelta(minutes=15 * 6)

    block_bars = infer_block_bars(
        timestamps=timestamps,
        label_end_timestamps=label_end,
        purge_bars=4,
        embargo_bars=5,
        bar_minutes=15,
        min_block_bars=3,
    )

    assert block_bars == 6


def test_fold_block_bootstrap_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    context_a = _multi_source_context(tmp_path / "a")
    context_b = _multi_source_context(tmp_path / "b")
    config = FoldBlockBootstrapConfig(replicates=3, block_bars=2, random_seed=11)

    FoldBlockBootstrap(config).run(context_a)
    FoldBlockBootstrap(config).run(context_b)

    starts_a = context_a.state["bootstrap_plans"][0].paths[0].block_start_positions
    starts_b = context_b.state["bootstrap_plans"][0].paths[0].block_start_positions
    assert starts_a == starts_b


def test_fold_block_bootstrap_records_train_only_manifest(tmp_path: Path) -> None:
    context = _multi_source_context(tmp_path)

    result = FoldBlockBootstrap(FoldBlockBootstrapConfig(replicates=2, block_bars=2)).run(context)

    assert result.passed is True
    assert result.summary["bootstrap_train_only"] is True
    assert result.summary["fold_count"] == 1
    assert result.summary["bootstrap_replicates"] == 2
