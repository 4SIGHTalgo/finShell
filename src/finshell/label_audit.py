from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from finshell.core import ComponentResult, PipelineComponent, PipelineContext
from finshell.ingestion import ColumnRoleMap


@dataclass(frozen=True, slots=True)
class LabelAuditConfig:
    min_labeled_rows: int = 30
    allowed_label_values: tuple[int, ...] = (-1, 0, 1)
    max_overlap_rate: float = 0.80
    data_key: str = "data"


class LabelAuditor(PipelineComponent):
    def __init__(self, config: LabelAuditConfig | None = None, *, name: str = "label_audit") -> None:
        super().__init__(name=name)
        self.config = config or LabelAuditConfig()

    def run(self, context: PipelineContext) -> ComponentResult:
        frame = context.state.get(self.config.data_key)
        roles = context.state.get("roles")
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"context.state[{self.config.data_key!r}] must contain a pandas DataFrame")
        if not isinstance(roles, ColumnRoleMap):
            raise ValueError("context.state['roles'] must contain a ColumnRoleMap")
        if not roles.label:
            raise ValueError("label audit requires roles.label")

        label_values = pd.to_numeric(frame[roles.label], errors="coerce")
        labeled = label_values.dropna()
        fail_reasons: list[str] = []
        if len(labeled) < int(self.config.min_labeled_rows):
            fail_reasons.append("insufficient_labeled_rows")
        allowed = set(float(value) for value in self.config.allowed_label_values)
        invalid_mask = label_values.notna() & ~label_values.astype(float).isin(allowed)
        if bool(invalid_mask.any()):
            fail_reasons.append("label_value_outside_allowed_set")

        observed_violations = _observed_timestamp_violations(frame, roles.timestamp)
        if observed_violations:
            fail_reasons.append("observed_timestamp_violation")

        overlap_rate = _interval_overlap_rate(frame, roles)
        if np.isfinite(overlap_rate) and overlap_rate > float(self.config.max_overlap_rate):
            fail_reasons.append("excessive_label_overlap")

        label_counts = {
            str(int(key) if float(key).is_integer() else key): int(value)
            for key, value in label_values.dropna().value_counts().sort_index().items()
        }
        feature_contract = _feature_contract(frame, roles)
        summary: dict[str, Any] = {
            "rows": int(len(frame)),
            "labeled_rows": int(len(labeled)),
            "label_column": roles.label,
            "allowed_label_values": list(self.config.allowed_label_values),
            "label_counts": label_counts,
            "observed_timestamp_violations": int(observed_violations),
            "interval_overlap_rate": overlap_rate,
            "feature_contract": feature_contract,
            "fail_reasons": sorted(set(fail_reasons)),
        }
        return ComponentResult(component=self.name, passed=not fail_reasons, summary=summary)


def _observed_timestamp_violations(frame: pd.DataFrame, timestamp_col: str) -> int:
    target_ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    violations = 0
    for column in frame.columns:
        if not str(column).endswith("_observed_timestamp_utc"):
            continue
        observed = pd.to_datetime(frame[column], utc=True, errors="coerce")
        violations += int((observed.notna() & target_ts.notna() & observed.gt(target_ts)).sum())
    return violations


def _interval_overlap_rate(frame: pd.DataFrame, roles: ColumnRoleMap) -> float:
    if not roles.label_end_timestamp:
        return float("nan")
    start = pd.to_datetime(frame[roles.timestamp], utc=True, errors="coerce")
    end = pd.to_datetime(frame[roles.label_end_timestamp], utc=True, errors="coerce")
    intervals = pd.DataFrame({"start": start, "end": end}).dropna().sort_values("start", kind="mergesort")
    if len(intervals) < 2:
        return 0.0
    previous_end = intervals["end"].shift(1)
    return float(intervals["start"].lt(previous_end).mean())


def _feature_contract(frame: pd.DataFrame, roles: ColumnRoleMap) -> list[dict[str, str]]:
    role_columns = {column: role for role, column in roles.named_columns().items()}
    rows: list[dict[str, str]] = []
    for column in frame.columns:
        name = str(column)
        if name in role_columns:
            role = f"role:{role_columns[name]}"
        elif name.endswith("_observed_timestamp_utc"):
            role = "availability_audit_do_not_use_as_feature"
        elif name.startswith("future_") or name.startswith("target_") or name.startswith("label_"):
            role = "leakage_risk_do_not_use_as_feature"
        elif name.endswith("_feature") or name.endswith("_value") or name.endswith("_threshold"):
            role = "candidate_feature"
        else:
            role = "unknown_review_before_feature_use"
        rows.append({"column": name, "role": role})
    return rows

