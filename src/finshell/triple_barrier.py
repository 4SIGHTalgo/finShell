from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from finshell.core import ComponentResult, PipelineComponent, PipelineContext
from finshell.ingestion import ColumnRoleMap


@dataclass(frozen=True, slots=True)
class TripleBarrierConfig:
    enabled: bool = True
    profit_take: float = 0.02
    stop_loss: float = 0.02
    vertical_bars: int = 24
    data_key: str = "development_data"
    output_key: str = "triple_barrier_result"

    def __post_init__(self) -> None:
        if self.profit_take <= 0:
            raise ValueError("profit_take must be > 0")
        if self.stop_loss <= 0:
            raise ValueError("stop_loss must be > 0")
        if self.vertical_bars < 1:
            raise ValueError("vertical_bars must be >= 1")


class TripleBarrierComparator(PipelineComponent):
    def __init__(self, config: TripleBarrierConfig | None = None, *, name: str = "triple_barrier") -> None:
        super().__init__(name=name)
        self.config = config or TripleBarrierConfig()

    def run(self, context: PipelineContext) -> ComponentResult:
        if not self.config.enabled:
            return ComponentResult(component=self.name, passed=True, summary={"skipped": True})
        frame = context.state.get(self.config.data_key)
        roles = context.state.get("roles")
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"context.state[{self.config.data_key!r}] must contain a pandas DataFrame")
        if not isinstance(roles, ColumnRoleMap):
            raise ValueError("context.state['roles'] must contain a ColumnRoleMap")
        for role_name, column in {"high": roles.high, "low": roles.low, "close": roles.close}.items():
            if not column:
                raise ValueError(f"triple barrier requires roles.{role_name}")
        result = _compute_triple_barrier(frame, roles, self.config)
        context.state[self.config.output_key] = result
        label_counts = {
            str(int(key)): int(value)
            for key, value in result["barrier_label"].value_counts().sort_index().items()
        }
        summary: dict[str, Any] = {
            "skipped": False,
            "rows": int(len(result)),
            "profit_take": float(self.config.profit_take),
            "stop_loss": float(self.config.stop_loss),
            "vertical_bars": int(self.config.vertical_bars),
            "label_counts": label_counts,
        }
        if roles.label and roles.label in frame.columns:
            supplied = pd.to_numeric(frame[roles.label], errors="coerce")
            valid = supplied.notna() & result["barrier_label"].notna()
            summary["agreement_rate"] = (
                float(supplied.loc[valid].astype(int).eq(result.loc[valid, "barrier_label"].astype(int)).mean())
                if bool(valid.any())
                else float("nan")
            )
        return ComponentResult(component=self.name, passed=True, summary=summary)


def _compute_triple_barrier(frame: pd.DataFrame, roles: ColumnRoleMap, config: TripleBarrierConfig) -> pd.DataFrame:
    working = frame.copy().reset_index(drop=True)
    high = pd.to_numeric(working[roles.high], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(working[roles.low], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(working[roles.close], errors="coerce").to_numpy(dtype=float)
    if roles.side and roles.side in working.columns:
        side = pd.to_numeric(working[roles.side], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        side = np.where(side < 0, -1.0, 1.0)
    else:
        side = np.ones(len(working), dtype=float)

    labels: list[int] = []
    reasons: list[str] = []
    exit_indices: list[int] = []
    for index in range(len(working)):
        entry = close[index]
        current_side = side[index]
        label = 0
        reason = "timeout"
        exit_index = min(len(working) - 1, index + int(config.vertical_bars))
        if not np.isfinite(entry):
            labels.append(0)
            reasons.append("invalid_entry")
            exit_indices.append(index)
            continue
        for path_index in range(index + 1, min(len(working), index + int(config.vertical_bars) + 1)):
            if current_side > 0:
                take_profit = high[path_index] >= entry * (1.0 + float(config.profit_take))
                stop_loss = low[path_index] <= entry * (1.0 - float(config.stop_loss))
            else:
                take_profit = low[path_index] <= entry * (1.0 - float(config.profit_take))
                stop_loss = high[path_index] >= entry * (1.0 + float(config.stop_loss))
            if take_profit:
                label = 1
                reason = "take_profit"
                exit_index = path_index
                break
            if stop_loss:
                label = -1
                reason = "stop_loss"
                exit_index = path_index
                break
        labels.append(label)
        reasons.append(reason)
        exit_indices.append(exit_index)
    return pd.DataFrame(
        {
            "barrier_label": labels,
            "exit_reason": reasons,
            "exit_index": exit_indices,
        }
    )

