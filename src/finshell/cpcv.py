from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import pandas as pd

from finshell.core import ComponentResult, PipelineComponent, PipelineContext
from finshell.ingestion import ColumnRoleMap


@dataclass(frozen=True, slots=True)
class CPCVConfig:
    n_groups: int = 6
    holdout_groups: int = 2
    validate_groups: int = 1
    purge_bars: int = 0
    embargo_bars: int = 0
    max_splits: int | None = None
    data_key: str = "development_data"

    def __post_init__(self) -> None:
        if self.n_groups < 3:
            raise ValueError("n_groups must be >= 3")
        if self.holdout_groups < 2:
            raise ValueError("holdout_groups must be >= 2")
        if self.holdout_groups >= self.n_groups:
            raise ValueError("holdout_groups must be < n_groups")
        if self.validate_groups < 1 or self.validate_groups >= self.holdout_groups:
            raise ValueError("validate_groups must be >= 1 and < holdout_groups")
        if min(self.purge_bars, self.embargo_bars) < 0:
            raise ValueError("purge_bars and embargo_bars must be >= 0")
        if self.max_splits is not None and self.max_splits < 1:
            raise ValueError("max_splits must be >= 1")


@dataclass(frozen=True, slots=True)
class CPCVFold:
    name: str
    validate_group_indices: list[int]
    test_group_indices: list[int]
    train_indices: list[int]
    validate_indices: list[int]
    test_indices: list[int]
    heldout_intervals: list[dict[str, str]]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "validate_group_indices": self.validate_group_indices,
            "test_group_indices": self.test_group_indices,
            "train_rows": len(self.train_indices),
            "validate_rows": len(self.validate_indices),
            "test_rows": len(self.test_indices),
            "heldout_intervals": self.heldout_intervals,
        }


class CPCVPurgeEmbargo(PipelineComponent):
    def __init__(self, config: CPCVConfig | None = None, *, name: str = "cpcv_purge_embargo") -> None:
        super().__init__(name=name)
        self.config = config or CPCVConfig()

    def run(self, context: PipelineContext) -> ComponentResult:
        frame = context.state.get(self.config.data_key)
        roles = context.state.get("roles")
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"context.state[{self.config.data_key!r}] must contain a pandas DataFrame")
        if not isinstance(roles, ColumnRoleMap):
            raise ValueError("context.state['roles'] must contain a ColumnRoleMap")
        working = frame.copy().reset_index(drop=True)
        working[roles.timestamp] = pd.to_datetime(working[roles.timestamp], utc=True, errors="coerce")
        if working[roles.timestamp].isna().any():
            raise ValueError(f"timestamp role column contains invalid timestamps: {roles.timestamp}")
        working = working.sort_values(roles.timestamp, kind="mergesort").reset_index(drop=True)
        groups = _chronological_groups(working, roles.timestamp, self.config.n_groups)
        folds: list[CPCVFold] = []
        for split_index, combo in enumerate(combinations(range(self.config.n_groups), self.config.holdout_groups), start=1):
            if self.config.max_splits is not None and len(folds) >= self.config.max_splits:
                break
            validate_group_ids = list(combo[: self.config.validate_groups])
            test_group_ids = list(combo[self.config.validate_groups :])
            validate_indices = _indices_for_groups(groups, validate_group_ids)
            test_indices = _indices_for_groups(groups, test_group_ids)
            heldout_indices = sorted(validate_indices + test_indices)
            excluded = set(heldout_indices)
            if heldout_indices:
                start = max(0, min(heldout_indices) - int(self.config.purge_bars))
                end = min(len(working) - 1, max(heldout_indices) + int(self.config.embargo_bars))
                excluded.update(range(start, end + 1))
            train_indices = [idx for idx in range(len(working)) if idx not in excluded]
            heldout_intervals = [
                {
                    "start": working.loc[min(_indices_for_groups(groups, [group_id])), roles.timestamp].isoformat(),
                    "end": working.loc[max(_indices_for_groups(groups, [group_id])), roles.timestamp].isoformat(),
                }
                for group_id in combo
            ]
            folds.append(
                CPCVFold(
                    name=f"cpcv_{split_index:03d}",
                    validate_group_indices=[idx + 1 for idx in validate_group_ids],
                    test_group_indices=[idx + 1 for idx in test_group_ids],
                    train_indices=train_indices,
                    validate_indices=validate_indices,
                    test_indices=test_indices,
                    heldout_intervals=heldout_intervals,
                )
            )
        manifest = {
            "cv_type": "combinatorial_purged_k_fold",
            "n_groups": int(self.config.n_groups),
            "holdout_groups": int(self.config.holdout_groups),
            "validate_groups": int(self.config.validate_groups),
            "purge_bars": int(self.config.purge_bars),
            "embargo_bars": int(self.config.embargo_bars),
            "fold_count": int(len(folds)),
            "folds": [fold.to_manifest() for fold in folds],
        }
        context.state["cpcv_folds"] = folds
        context.state["cpcv_manifest"] = manifest
        return ComponentResult(component=self.name, passed=bool(folds), summary=manifest)


def _chronological_groups(frame: pd.DataFrame, timestamp_col: str, n_groups: int) -> list[list[int]]:
    if len(frame) < n_groups:
        raise ValueError(f"need at least {n_groups} rows for CPCV; got {len(frame)}")
    base = len(frame) // n_groups
    remainder = len(frame) % n_groups
    groups: list[list[int]] = []
    cursor = 0
    for index in range(n_groups):
        size = base + (1 if index < remainder else 0)
        groups.append(list(range(cursor, cursor + size)))
        cursor += size
    return groups


def _indices_for_groups(groups: list[list[int]], group_ids: list[int]) -> list[int]:
    out: list[int] = []
    for group_id in group_ids:
        out.extend(groups[group_id])
    return sorted(out)

