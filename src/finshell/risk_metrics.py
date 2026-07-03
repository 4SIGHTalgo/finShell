from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from finshell.core import ComponentResult, PipelineComponent, PipelineContext
from finshell.ingestion import ColumnRoleMap


@dataclass(frozen=True, slots=True)
class RiskMetricsConfig:
    data_key: str = "development_data"
    min_selected: int = 1
    annualization_factor: float = 1.0
    cdar_quantile: float = 0.95

    def __post_init__(self) -> None:
        if self.min_selected < 1:
            raise ValueError("min_selected must be >= 1")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be > 0")
        if not 0.0 < self.cdar_quantile <= 1.0:
            raise ValueError("cdar_quantile must be in (0, 1]")


class RiskMetrics(PipelineComponent):
    def __init__(self, config: RiskMetricsConfig | None = None, *, name: str = "risk_metrics") -> None:
        super().__init__(name=name)
        self.config = config or RiskMetricsConfig()

    def run(self, context: PipelineContext) -> ComponentResult:
        frame = context.state.get(self.config.data_key)
        roles = context.state.get("roles")
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"context.state[{self.config.data_key!r}] must contain a pandas DataFrame")
        if not isinstance(roles, ColumnRoleMap):
            raise ValueError("context.state['roles'] must contain a ColumnRoleMap")
        if not roles.outcome:
            return ComponentResult(
                component=self.name,
                passed=False,
                summary={"fail_reasons": ["missing_outcome_role"]},
            )
        outcomes = pd.to_numeric(frame[roles.outcome], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if roles.selected and roles.selected in frame.columns:
            selected = frame[roles.selected].astype(bool)
            values = outcomes.loc[selected].dropna().to_numpy(dtype=float)
        else:
            values = outcomes.dropna().to_numpy(dtype=float)
        summary = compute_risk_metrics(
            values,
            min_selected=int(self.config.min_selected),
            annualization_factor=float(self.config.annualization_factor),
            cdar_quantile=float(self.config.cdar_quantile),
        )
        context.state["risk_metrics_summary"] = summary
        return ComponentResult(component=self.name, passed=not summary["fail_reasons"], summary=summary)


def compute_risk_metrics(
    values: np.ndarray,
    *,
    min_selected: int,
    annualization_factor: float,
    cdar_quantile: float,
) -> dict[str, Any]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    fail_reasons: list[str] = []
    selected_count = int(clean.size)
    if selected_count < int(min_selected):
        fail_reasons.append("insufficient_selected_count")
    mean_return = float(np.mean(clean)) if selected_count else float("nan")
    volatility = float(np.std(clean, ddof=1)) if selected_count > 1 else 0.0
    downside = clean[clean < 0.0]
    downside_vol = float(np.std(downside, ddof=1)) if downside.size > 1 else (float(abs(downside[0])) if downside.size == 1 else 0.0)
    scale = float(np.sqrt(annualization_factor))
    sharpe_like = float(mean_return / volatility * scale) if volatility > 0.0 else float("nan")
    sortino_like = float(mean_return / downside_vol * scale) if downside_vol > 0.0 else float("nan")
    equity = np.cumsum(clean) if selected_count else np.asarray([], dtype=float)
    if equity.size:
        running_peak = np.maximum.accumulate(equity)
        drawdowns = running_peak - equity
        max_drawdown = float(np.max(drawdowns))
        cutoff = float(np.quantile(drawdowns, cdar_quantile))
        cdar_like = float(np.mean(drawdowns[drawdowns >= cutoff])) if np.any(drawdowns >= cutoff) else 0.0
    else:
        max_drawdown = float("nan")
        cdar_like = float("nan")
    hit_rate = float(np.mean(clean > 0.0)) if selected_count else float("nan")
    return {
        "selected_count": selected_count,
        "mean_return": mean_return,
        "median_return": float(np.median(clean)) if selected_count else float("nan"),
        "total_return": float(np.sum(clean)) if selected_count else 0.0,
        "volatility": volatility,
        "sharpe_like": sharpe_like,
        "sortino_like": sortino_like,
        "max_drawdown": max_drawdown,
        "cdar_like": cdar_like,
        "hit_rate": hit_rate,
        "fail_reasons": fail_reasons,
    }

