from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from finshell.core import ComponentResult, PipelineComponent, PipelineContext
from finshell.ingestion import ColumnRoleMap


@dataclass(frozen=True, slots=True)
class NullTestConfig:
    random_simulations: int = 512
    random_seed: int = 42
    min_random_percentile: float = 0.95
    max_p_value: float = 0.05
    data_key: str = "development_data"

    def __post_init__(self) -> None:
        if self.random_simulations < 1:
            raise ValueError("random_simulations must be >= 1")
        if not 0.0 < self.min_random_percentile <= 1.0:
            raise ValueError("min_random_percentile must be in (0, 1]")
        if not 0.0 <= self.max_p_value <= 1.0:
            raise ValueError("max_p_value must be in [0, 1]")


class NullTestSuite(PipelineComponent):
    def __init__(self, config: NullTestConfig | None = None, *, name: str = "null_tests") -> None:
        super().__init__(name=name)
        self.config = config or NullTestConfig()

    def run(self, context: PipelineContext) -> ComponentResult:
        frame = context.state.get(self.config.data_key)
        roles = context.state.get("roles")
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"context.state[{self.config.data_key!r}] must contain a pandas DataFrame")
        if not isinstance(roles, ColumnRoleMap):
            raise ValueError("context.state['roles'] must contain a ColumnRoleMap")
        if not roles.selected or not roles.outcome:
            return ComponentResult(
                component=self.name,
                passed=False,
                summary={"fail_reasons": ["missing_selected_or_outcome_role"]},
            )

        selected_mask = frame[roles.selected].astype(bool).to_numpy()
        outcomes = pd.to_numeric(frame[roles.outcome], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = outcomes.notna().to_numpy()
        selected_mask = selected_mask & valid
        values = outcomes.loc[valid].to_numpy(dtype=float)
        valid_selected = selected_mask[valid]
        selected_count = int(valid_selected.sum())
        observed_total = float(values[valid_selected].sum()) if selected_count else 0.0
        fail_reasons: list[str] = []
        if selected_count <= 0:
            fail_reasons.append("no_selected_rows")
            random_totals = np.asarray([], dtype=float)
        else:
            rng = np.random.default_rng(int(self.config.random_seed))
            random_totals = np.asarray(
                [
                    float(values[rng.choice(len(values), size=selected_count, replace=False)].sum())
                    for _ in range(int(self.config.random_simulations))
                ],
                dtype=float,
            )
        if random_totals.size:
            random_p95 = float(np.quantile(random_totals, 0.95))
            percentile = float(np.mean(random_totals <= observed_total))
            p_value = float(np.mean(random_totals >= observed_total))
        else:
            random_p95 = float("nan")
            percentile = float("nan")
            p_value = float("nan")
        if np.isfinite(random_p95) and observed_total <= random_p95:
            fail_reasons.append("real_path_not_above_random_p95")
        if np.isfinite(percentile) and percentile < float(self.config.min_random_percentile):
            fail_reasons.append("random_percentile_below_threshold")
        if np.isfinite(p_value) and p_value > float(self.config.max_p_value):
            fail_reasons.append("p_value_above_threshold")
        summary: dict[str, Any] = {
            "rows": int(len(frame)),
            "selected_count": selected_count,
            "observed_total": observed_total,
            "same_count_random_p95": random_p95,
            "observed_random_percentile": percentile,
            "p_value": p_value,
            "random_simulations": int(self.config.random_simulations),
            "random_totals": [float(value) for value in random_totals.tolist()],
            "fail_reasons": sorted(set(fail_reasons)),
        }
        context.state["null_test_summary"] = summary
        return ComponentResult(component=self.name, passed=not fail_reasons, summary=summary)

