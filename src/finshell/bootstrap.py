from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from finshell.core import ComponentResult, PipelineComponent, PipelineContext
from finshell.cpcv import CPCVFold
from finshell.ingestion import ColumnRoleMap

BOOTSTRAP_CONTRACT_VERSION = "fold_block_bootstrap_v1"


@dataclass(frozen=True, slots=True)
class FoldBlockBootstrapConfig:
    replicates: int = 8
    block_bars: int | None = None
    random_seed: int = 42
    min_block_bars: int = 8
    data_key: str = "development_data"

    def __post_init__(self) -> None:
        if self.replicates < 1:
            raise ValueError("replicates must be >= 1")
        if self.block_bars is not None and self.block_bars < 1:
            raise ValueError("block_bars must be >= 1")
        if self.min_block_bars < 1:
            raise ValueError("min_block_bars must be >= 1")


@dataclass(frozen=True, slots=True)
class BootstrapPath:
    replicate: int
    indices: list[int]
    timestamp_ns: list[int]
    block_start_positions: list[int]


@dataclass(frozen=True, slots=True)
class FoldBootstrapPlan:
    fold_name: str
    contract_version: str
    sampling_unit: str
    synchronized_sources: bool
    train_only: bool
    block_bars: int
    paths: list[BootstrapPath]

    def resample(self, frame: pd.DataFrame, path: BootstrapPath) -> pd.DataFrame:
        return frame.iloc[path.indices].copy().reset_index(drop=True)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "fold_name": self.fold_name,
            "contract_version": self.contract_version,
            "sampling_unit": self.sampling_unit,
            "synchronized_sources": self.synchronized_sources,
            "train_only": self.train_only,
            "block_bars": self.block_bars,
            "replicates": len(self.paths),
        }


class FoldBlockBootstrap(PipelineComponent):
    def __init__(self, config: FoldBlockBootstrapConfig | None = None, *, name: str = "fold_block_bootstrap") -> None:
        super().__init__(name=name)
        self.config = config or FoldBlockBootstrapConfig()

    def run(self, context: PipelineContext) -> ComponentResult:
        frame = context.state.get(self.config.data_key)
        roles = context.state.get("roles")
        folds = context.state.get("cpcv_folds")
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"context.state[{self.config.data_key!r}] must contain a pandas DataFrame")
        if not isinstance(roles, ColumnRoleMap):
            raise ValueError("context.state['roles'] must contain a ColumnRoleMap")
        if not isinstance(folds, list) or not all(isinstance(item, CPCVFold) for item in folds):
            raise ValueError("context.state['cpcv_folds'] must contain CPCVFold objects")
        plans = [
            _build_plan_for_fold(frame=frame, fold=fold, roles=roles, config=self.config, fold_offset=index)
            for index, fold in enumerate(folds)
        ]
        context.state["bootstrap_plans"] = plans
        summary = {
            "bootstrap_contract_version": BOOTSTRAP_CONTRACT_VERSION,
            "bootstrap_train_only": True,
            "bootstrap_sampling_unit": "timestamp_block",
            "bootstrap_synchronized_sources": True,
            "bootstrap_replicates": int(self.config.replicates),
            "fold_count": int(len(plans)),
            "folds": [plan.to_manifest() for plan in plans],
        }
        context.state["bootstrap_manifest"] = summary
        return ComponentResult(component=self.name, passed=bool(plans), summary=summary)


def infer_block_bars(
    *,
    timestamps: pd.Series | list[Any],
    label_end_timestamps: pd.Series | list[Any] | None = None,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    bar_minutes: int = 1,
    min_block_bars: int = 8,
) -> int:
    candidates = [int(min_block_bars), int(purge_bars), int(embargo_bars)]
    if label_end_timestamps is not None:
        start = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
        end = pd.to_datetime(pd.Series(label_end_timestamps), utc=True, errors="coerce")
        bars = np.ceil((end - start).dt.total_seconds() / 60.0 / max(1, int(bar_minutes)))
        finite = bars[np.isfinite(bars) & (bars > 0)]
        if not finite.empty:
            candidates.append(int(finite.max()))
    return max(1, max(candidates))


def _build_plan_for_fold(
    *,
    frame: pd.DataFrame,
    fold: CPCVFold,
    roles: ColumnRoleMap,
    config: FoldBlockBootstrapConfig,
    fold_offset: int,
) -> FoldBootstrapPlan:
    train = frame.iloc[fold.train_indices].copy()
    timestamps = pd.to_datetime(train[roles.timestamp], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"timestamp role column contains invalid timestamps: {roles.timestamp}")
    unique_ns = list(dict.fromkeys(timestamps.astype("int64").tolist()))
    positions_by_timestamp: dict[int, list[int]] = {int(value): [] for value in unique_ns}
    for frame_index, value in zip(train.index.tolist(), timestamps.astype("int64").tolist()):
        positions_by_timestamp[int(value)].append(int(frame_index))
    block_bars = min(int(config.block_bars or config.min_block_bars), len(unique_ns))
    rng = np.random.default_rng(int(config.random_seed) + int(fold_offset))
    paths: list[BootstrapPath] = []
    for replicate in range(int(config.replicates)):
        sampled_timestamps: list[int] = []
        sampled_indices: list[int] = []
        starts: list[int] = []
        while len(sampled_timestamps) < len(unique_ns):
            start = int(rng.integers(0, len(unique_ns)))
            starts.append(start)
            for timestamp in unique_ns[start : min(len(unique_ns), start + block_bars)]:
                sampled_timestamps.append(int(timestamp))
                sampled_indices.extend(positions_by_timestamp[int(timestamp)])
                if len(sampled_timestamps) >= len(unique_ns):
                    break
        paths.append(
            BootstrapPath(
                replicate=int(replicate),
                indices=sampled_indices,
                timestamp_ns=sampled_timestamps,
                block_start_positions=starts,
            )
        )
    return FoldBootstrapPlan(
        fold_name=fold.name,
        contract_version=BOOTSTRAP_CONTRACT_VERSION,
        sampling_unit="timestamp_block",
        synchronized_sources=True,
        train_only=True,
        block_bars=int(block_bars),
        paths=paths,
    )

