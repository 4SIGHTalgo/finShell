from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from finshell.core import ComponentResult, PipelineComponent, PipelineContext
from finshell.ingestion import ColumnRoleMap


@dataclass(frozen=True, slots=True)
class HoldoutConfig:
    quarantine_fraction: float = 0.20
    quarantine_start: str | None = None
    quarantine_end_exclusive: str | None = None
    data_key: str = "data"
    development_key: str = "development_data"
    quarantine_key: str = "quarantine_data"

    def __post_init__(self) -> None:
        if not 0.0 < float(self.quarantine_fraction) < 1.0:
            raise ValueError("quarantine_fraction must be between 0 and 1")
        if (self.quarantine_start is None) != (self.quarantine_end_exclusive is None):
            raise ValueError("quarantine_start and quarantine_end_exclusive must be provided together")


class HoldoutSplitter(PipelineComponent):
    def __init__(self, config: HoldoutConfig | None = None, *, name: str = "holdout_split") -> None:
        super().__init__(name=name)
        self.config = config or HoldoutConfig()

    def run(self, context: PipelineContext) -> ComponentResult:
        frame = context.state.get(self.config.data_key)
        roles = context.state.get("roles")
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"context.state[{self.config.data_key!r}] must contain a pandas DataFrame")
        if not isinstance(roles, ColumnRoleMap):
            raise ValueError("context.state['roles'] must contain a ColumnRoleMap")
        working = frame.copy()
        working[roles.timestamp] = pd.to_datetime(working[roles.timestamp], utc=True, errors="coerce")
        if working[roles.timestamp].isna().any():
            raise ValueError(f"timestamp role column contains invalid timestamps: {roles.timestamp}")
        working = working.sort_values(roles.timestamp, kind="mergesort").reset_index(drop=True)

        if self.config.quarantine_start and self.config.quarantine_end_exclusive:
            start = pd.Timestamp(self.config.quarantine_start)
            end = pd.Timestamp(self.config.quarantine_end_exclusive)
            mask = working[roles.timestamp].ge(start) & working[roles.timestamp].lt(end)
            mode = "explicit_window"
        else:
            holdout_rows = max(1, int(round(len(working) * float(self.config.quarantine_fraction))))
            split_at = max(0, len(working) - holdout_rows)
            mask = pd.Series(False, index=working.index)
            mask.iloc[split_at:] = True
            mode = "fraction_tail"

        development = working.loc[~mask].copy().reset_index(drop=True)
        quarantine = working.loc[mask].copy().reset_index(drop=True)
        context.state[self.config.development_key] = development
        context.state[self.config.quarantine_key] = quarantine
        context.state["sealed_partitions"] = ["quarantine_holdout"]
        summary: dict[str, Any] = {
            "mode": mode,
            "timestamp_col": roles.timestamp,
            "input_rows": int(len(working)),
            "development_rows": int(len(development)),
            "quarantine_rows": int(len(quarantine)),
            "quarantine_fraction": float(self.config.quarantine_fraction),
            "sealed_partitions": ["quarantine_holdout"],
        }
        return ComponentResult(component=self.name, passed=not quarantine.empty, summary=summary)

