from __future__ import annotations

from pathlib import Path

import pandas as pd

from finshell.core import PipelineContext
from finshell.pbo import PBOAnalyzer, PBOConfig, binomial_upper_confidence_bound


def test_pbo_flags_validate_winner_that_fails_test_rank(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        [
            {"fold": "f1", "partition": "validate", "candidate": "a", "total_return": 10.0},
            {"fold": "f1", "partition": "validate", "candidate": "b", "total_return": 5.0},
            {"fold": "f1", "partition": "test", "candidate": "a", "total_return": -1.0},
            {"fold": "f1", "partition": "test", "candidate": "b", "total_return": 2.0},
            {"fold": "f2", "partition": "validate", "candidate": "a", "total_return": 10.0},
            {"fold": "f2", "partition": "validate", "candidate": "b", "total_return": 5.0},
            {"fold": "f2", "partition": "test", "candidate": "a", "total_return": -2.0},
            {"fold": "f2", "partition": "test", "candidate": "b", "total_return": 3.0},
        ]
    )
    context = PipelineContext(tmp_path)
    context.state["candidate_metrics"] = metrics

    result = PBOAnalyzer(PBOConfig(min_folds=2)).run(context)

    assert result.passed is False
    assert result.summary["probability_of_backtest_overfit"] == 1.0
    assert result.summary["detail"][0]["overfit_fold"] is True


def test_pbo_supports_configurable_score_column(tmp_path: Path) -> None:
    context = PipelineContext(tmp_path)
    context.state["candidate_metrics"] = pd.DataFrame(
        [
            {"fold": "f1", "partition": "validate", "candidate": "a", "sharpe": 2.0},
            {"fold": "f1", "partition": "validate", "candidate": "b", "sharpe": 1.0},
            {"fold": "f1", "partition": "test", "candidate": "a", "sharpe": 2.0},
            {"fold": "f1", "partition": "test", "candidate": "b", "sharpe": 1.0},
        ]
    )

    result = PBOAnalyzer(PBOConfig(score_column="sharpe", min_folds=1, max_upper_confidence_bound=1.0)).run(context)

    assert result.passed is True
    assert result.summary["score_column"] == "sharpe"


def test_zero_observed_pbo_can_fail_closed_when_detection_power_is_too_low(tmp_path: Path) -> None:
    assert binomial_upper_confidence_bound(0, 15, confidence_level=0.95) > 0.10
    metrics = pd.DataFrame(
        [
            {"fold": f"f{fold}", "partition": partition, "candidate": candidate, "total_return": score}
            for fold in range(15)
            for candidate, score in (("a", 2.0), ("b", 1.0))
            for partition in ("validate", "test")
        ]
    )
    context = PipelineContext(tmp_path)
    context.state["candidate_metrics"] = metrics

    result = PBOAnalyzer(PBOConfig(min_folds=3, max_upper_confidence_bound=0.10)).run(context)

    assert result.passed is False
    assert result.summary["probability_of_backtest_overfit"] == 0.0
    assert "pbo_upper_confidence_above_limit" in result.summary["fail_reasons"]
