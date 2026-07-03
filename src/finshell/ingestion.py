from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from finshell.core import ComponentResult, PipelineComponent, PipelineContext


@dataclass(frozen=True, slots=True)
class ColumnRoleMap:
    timestamp: str
    label: str | None = None
    selected: str | None = None
    outcome: str | None = None
    side: str | None = None
    open: str | None = None
    high: str | None = None
    low: str | None = None
    close: str | None = None
    group: str | None = None
    label_end_timestamp: str | None = None

    def required_columns(self) -> dict[str, str]:
        required = {"timestamp": self.timestamp}
        if self.label:
            required["label"] = self.label
        return required

    def named_columns(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if value:
                out[item.name] = str(value)
        return out

    def validate(self, frame: pd.DataFrame) -> None:
        missing = [
            f"{role}={column}"
            for role, column in self.required_columns().items()
            if column not in frame.columns
        ]
        if missing:
            raise ValueError("missing required role columns: " + ", ".join(missing))


@dataclass(frozen=True, slots=True)
class DataIngestConfig:
    source: pd.DataFrame | str | Path
    roles: ColumnRoleMap
    sort_by_timestamp: bool = True
    state_key: str = "data"


class DataIngestor(PipelineComponent):
    def __init__(self, config: DataIngestConfig, *, name: str = "data_ingestion") -> None:
        super().__init__(name=name)
        self.config = config

    def run(self, context: PipelineContext) -> ComponentResult:
        frame, source_format = self._load_frame(self.config.source)
        roles = self.config.roles
        roles.validate(frame)
        working = frame.copy()
        timestamp_col = roles.timestamp
        working[timestamp_col] = pd.to_datetime(working[timestamp_col], utc=True, errors="coerce")
        if working[timestamp_col].isna().any():
            raise ValueError(f"timestamp role column contains invalid timestamps: {timestamp_col}")
        if self.config.sort_by_timestamp:
            working = working.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
        context.state[self.config.state_key] = working
        context.state["roles"] = roles
        timestamps = working[timestamp_col]
        summary: dict[str, Any] = {
            "source_format": source_format,
            "rows": int(len(working)),
            "columns": list(map(str, working.columns)),
            "roles": roles.named_columns(),
            "timestamp_col": timestamp_col,
            "timestamp_min_utc": timestamps.min().isoformat() if not timestamps.empty else None,
            "timestamp_max_utc": timestamps.max().isoformat() if not timestamps.empty else None,
        }
        return ComponentResult(component=self.name, passed=True, summary=summary)

    @staticmethod
    def _load_frame(source: pd.DataFrame | str | Path) -> tuple[pd.DataFrame, str]:
        if isinstance(source, pd.DataFrame):
            return source.copy(), "dataframe"
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path), "csv"
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path), "parquet"
        raise ValueError(f"unsupported data source format: {path}")

