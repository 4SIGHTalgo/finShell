from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from finshell.core import ComponentResult, PipelineComponent, PipelineContext


@dataclass(frozen=True, slots=True)
class PBOConfig:
    metrics_key: str = "candidate_metrics"
    fold_column: str = "fold"
    partition_column: str = "partition"
    candidate_column: str = "candidate"
    score_column: str = "total_return"
    validate_partition: str = "validate"
    test_partition: str = "test"
    min_folds: int = 3
    min_candidates: int = 2
    max_pbo: float = 0.10
    confidence_level: float = 0.95
    max_upper_confidence_bound: float = 0.10


class PBOAnalyzer(PipelineComponent):
    def __init__(self, config: PBOConfig | None = None, *, name: str = "pbo") -> None:
        super().__init__(name=name)
        self.config = config or PBOConfig()

    def run(self, context: PipelineContext) -> ComponentResult:
        metrics = context.state.get(self.config.metrics_key)
        if not isinstance(metrics, pd.DataFrame):
            raise ValueError(f"context.state[{self.config.metrics_key!r}] must contain a pandas DataFrame")
        summary = probability_of_backtest_overfit(metrics, self.config)
        context.state["pbo_summary"] = summary
        return ComponentResult(component=self.name, passed=not summary["fail_reasons"], summary=summary)


def probability_of_backtest_overfit(metrics: pd.DataFrame, config: PBOConfig) -> dict[str, Any]:
    required = {
        config.fold_column,
        config.partition_column,
        config.candidate_column,
        config.score_column,
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        return _fail_summary(config, [f"missing_columns:{','.join(missing)}"], [])

    detail: list[dict[str, Any]] = []
    for fold, group in metrics.groupby(config.fold_column, sort=True):
        validate = group.loc[group[config.partition_column].astype(str).eq(config.validate_partition)]
        test = group.loc[group[config.partition_column].astype(str).eq(config.test_partition)]
        if validate.empty or test.empty:
            continue
        validate_scores = pd.to_numeric(validate[config.score_column], errors="coerce")
        if validate_scores.notna().sum() < 2:
            continue
        winner_idx = validate_scores.idxmax()
        winner = validate.loc[winner_idx]
        winner_candidate = winner[config.candidate_column]
        test_scores = (
            test.set_index(config.candidate_column)[config.score_column]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        if winner_candidate not in test_scores.index or len(test_scores) < 2:
            continue
        winner_test_score = float(test_scores.loc[winner_candidate])
        rank_pct = float((test_scores <= winner_test_score).mean())
        rank_pct = min(max(rank_pct, 1e-12), 1.0 - 1e-12)
        detail.append(
            {
                "fold": str(fold),
                "selected_candidate": str(winner_candidate),
                "validate_score": float(winner[config.score_column]),
                "test_score": winner_test_score,
                "test_rank_percentile": rank_pct,
                "logit_test_rank": math.log(rank_pct / (1.0 - rank_pct)),
                "overfit_fold": bool(rank_pct <= 0.50),
            }
        )

    if not detail:
        return _fail_summary(config, ["insufficient_validate_test_pairs"], detail)

    fold_count = len(detail)
    overfit_count = int(sum(1 for row in detail if row["overfit_fold"]))
    pbo = float(overfit_count / fold_count)
    candidate_count = int(metrics[config.candidate_column].nunique())
    upper_bound = binomial_upper_confidence_bound(
        overfit_count,
        fold_count,
        confidence_level=float(config.confidence_level),
    )
    fail_reasons: list[str] = []
    if pbo > float(config.max_pbo):
        fail_reasons.append("probability_of_backtest_overfit_above_limit")
    if fold_count < int(config.min_folds):
        fail_reasons.append("insufficient_pbo_folds")
    if candidate_count < int(config.min_candidates):
        fail_reasons.append("insufficient_pbo_candidates")
    if not np.isfinite(upper_bound) or upper_bound > float(config.max_upper_confidence_bound):
        fail_reasons.append("pbo_upper_confidence_above_limit")
    return {
        "score_column": config.score_column,
        "fold_count": int(fold_count),
        "candidate_count": int(candidate_count),
        "overfit_fold_count": int(overfit_count),
        "probability_of_backtest_overfit": pbo,
        "pbo_upper_confidence_bound": upper_bound,
        "fail_reasons": sorted(set(fail_reasons)),
        "detail": detail,
    }


def binomial_upper_confidence_bound(successes: int, trials: int, *, confidence_level: float) -> float:
    trials = int(trials)
    successes = int(successes)
    if trials <= 0 or successes < 0 or successes > trials:
        return float("nan")
    alpha = 1.0 - min(max(float(confidence_level), 0.0), 1.0)
    if alpha <= 0.0:
        return 1.0
    if successes >= trials:
        return 1.0
    lo = successes / trials
    hi = 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        cdf = _binomial_cdf(successes, trials, mid)
        if cdf > alpha:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    probability = min(max(float(probability), 0.0), 1.0)
    total = 0.0
    for index in range(int(successes) + 1):
        total += math.comb(int(trials), int(index)) * (probability**index) * (
            (1.0 - probability) ** (int(trials) - int(index))
        )
    return float(total)


def _fail_summary(config: PBOConfig, fail_reasons: list[str], detail: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "score_column": config.score_column,
        "fold_count": len(detail),
        "candidate_count": 0,
        "overfit_fold_count": 0,
        "probability_of_backtest_overfit": float("nan"),
        "pbo_upper_confidence_bound": float("nan"),
        "fail_reasons": fail_reasons,
        "detail": detail,
    }

