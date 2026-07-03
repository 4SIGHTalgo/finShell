# finShell

Python validation engine for model-free financial label and backtest audits.

## Installation

Install the core package from a local checkout:

```powershell
python -m pip install .
```

`finShell` is built around a role-mapped, OOP pipeline:

1. ingest user data without hardcoded column names
2. audit label and feature contracts
3. seal a final quarantine holdout
4. build CPCV purge/embargo folds
5. bootstrap fold train paths
6. run selected-trade null tests
7. compute risk-adjusted path metrics

Plotting support is included in the normal installation. Figure generation is
opt-in and writes files only when a `PlotConfig` with `enabled=True` is supplied.

## Basic Pipeline

```python
import pandas as pd

from finshell import ColumnRoleMap, FullPipeline
from finshell.cpcv import CPCVConfig
from finshell.label_audit import LabelAuditConfig
from finshell.null_tests import NullTestConfig

frame = pd.read_csv("labels.csv")
roles = ColumnRoleMap(
    timestamp="event_time",
    label="triple_barrier_label",
    selected="selected",
    outcome="net_return",
)

pipeline = FullPipeline.validation_pipeline(
    source=frame,
    roles=roles,
    label_audit=LabelAuditConfig(allowed_label_values=(-1, 0, 1)),
    cpcv=CPCVConfig(n_groups=6, holdout_groups=2, validate_groups=1),
    null_tests=NullTestConfig(random_simulations=1000),
)

result = pipeline.run()
print(result.passed)
print(result.manifest)
```

## Stage Plots

Map OHLC columns when using the optional triple-barrier comparison:

```python
from finshell import ColumnRoleMap, FullPipeline, PlotConfig
from finshell.triple_barrier import TripleBarrierConfig

roles = ColumnRoleMap(
    timestamp="event_time",
    label="triple_barrier_label",
    selected="selected",
    outcome="net_return",
    high="high",
    low="low",
    close="close",
    side="side",
)

pipeline = FullPipeline.validation_pipeline(
    source="labels.parquet",
    roles=roles,
    artifact_dir="outputs/validation_run",
    plots=PlotConfig(
        enabled=True,
        max_paths=100,
        bootstrap_block_bars=8,
        include_fold_bootstrap=True,
        random_seed=42,
    ),
    triple_barrier=TripleBarrierConfig(
        profit_take=0.02,
        stop_loss=0.01,
        vertical_bars=24,
    ),
)

result = pipeline.run()
```

## Complete Label-to-Promotion Example

This executable example uses a triple-barrier path label on deterministic
synthetic OHLC data. `DummyModel` does not train or read labels, outcomes, or
future prices. It maps one decision-time feature to a score so the example can
focus on the validation API.

The promotion logic is ordinary user code evaluated after finShell finishes.
It is not a built-in promotion policy. The passing result demonstrates API
behavior on constructed data and is not evidence of a tradable edge.

<!-- full-pipeline-example:start -->
```python
from pathlib import Path

import numpy as np
import pandas as pd

from finshell import ColumnRoleMap, FullPipeline, PipelineContext, PlotConfig
from finshell.bootstrap import FoldBlockBootstrapConfig
from finshell.cpcv import CPCVConfig
from finshell.holdout import HoldoutConfig
from finshell.label_audit import LabelAuditConfig
from finshell.null_tests import NullTestConfig
from finshell.pbo import PBOAnalyzer, PBOConfig
from finshell.risk_metrics import RiskMetricsConfig
from finshell.triple_barrier import TripleBarrierComparator, TripleBarrierConfig


TRIPLE_BARRIER = TripleBarrierConfig(
    profit_take=0.01,
    stop_loss=0.01,
    vertical_bars=1,
)

promotion_gates = {
    "min_selected_trades": 30,
    "max_null_p_value": 0.05,
    "min_random_percentile": 0.95,
    "max_pbo": 0.10,
    "min_mean_return": 0.001,
    "max_drawdown": 0.25,
    "min_quarantine_mean_ratio": 0.50,
}


class DummyModel:
    """Score mapper only; it intentionally has no training method."""

    def __init__(self, *, invert: bool = False) -> None:
        self.invert = invert

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        feature = frame["signal_feature"].to_numpy(dtype=float)
        scores = 1.0 / (1.0 + np.exp(-3.0 * feature))
        return 1.0 - scores if self.invert else scores


def make_triple_barrier_frame(rows: int = 360) -> pd.DataFrame:
    # The feature at row t deterministically influences the synthetic move t+1.
    signal = np.sin(np.linspace(0.0, 18.0 * np.pi, rows))
    close = np.empty(rows, dtype=float)
    close[0] = 100.0
    for index in range(rows - 1):
        close[index + 1] = close[index] * (1.0 + 0.012 * signal[index])

    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC"),
            "signal_feature": signal,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "side": np.ones(rows, dtype=int),
        }
    )
    label_roles = ColumnRoleMap(
        timestamp="event_time",
        high="high",
        low="low",
        close="close",
        side="side",
    )
    label_context = PipelineContext(Path("outputs") / "label_generation")
    label_context.state["development_data"] = frame
    label_context.state["roles"] = label_roles
    TripleBarrierComparator(TRIPLE_BARRIER).run(label_context)
    barrier = label_context.state["triple_barrier_result"]
    frame["triple_barrier_label"] = barrier["barrier_label"].to_numpy()
    frame["economic_return"] = barrier["barrier_return"].to_numpy()
    return frame


def build_pbo_metrics(context: PipelineContext, thresholds: tuple[float, ...]) -> pd.DataFrame:
    development = context.state["development_data"]
    rows = []
    for fold in context.state["cpcv_folds"]:
        for partition, indices in (
            ("validate", fold.validate_indices),
            ("test", fold.test_indices),
        ):
            partition_frame = development.iloc[indices]
            for threshold in thresholds:
                selected = partition_frame["model_score"].ge(threshold)
                rows.append(
                    {
                        "fold": fold.name,
                        "partition": partition,
                        "candidate": f"threshold_{threshold:.2f}",
                        "total_return": float(
                            partition_frame.loc[selected, "economic_return"].sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def selected_mean(frame: pd.DataFrame) -> float:
    values = frame.loc[frame["selected"], "economic_return"].dropna()
    return float(values.mean()) if not values.empty else float("nan")


def evaluate_promotion(run_result, pbo_result, gates: dict[str, float]) -> dict:
    context = run_result.context
    null_summary = context.state["null_test_summary"]
    risk_summary = context.state["risk_metrics_summary"]
    development_mean = selected_mean(context.state["development_data"])
    quarantine_mean = selected_mean(context.state["quarantine_data"])
    pbo = pbo_result.summary["probability_of_backtest_overfit"]

    checks = {
        "pipeline_passed": run_result.passed,
        "selected_trade_count": (
            risk_summary["selected_count"] >= gates["min_selected_trades"]
        ),
        "null_p_value": null_summary["p_value"] <= gates["max_null_p_value"],
        "random_percentile": (
            null_summary["observed_random_percentile"]
            >= gates["min_random_percentile"]
        ),
        "pbo": pbo_result.passed and pbo <= gates["max_pbo"],
        "mean_return": (
            np.isfinite(development_mean)
            and development_mean >= gates["min_mean_return"]
        ),
        "max_drawdown": (
            np.isfinite(risk_summary["max_drawdown"])
            and risk_summary["max_drawdown"] <= gates["max_drawdown"]
        ),
        "quarantine_degradation": (
            np.isfinite(development_mean)
            and development_mean > 0.0
            and np.isfinite(quarantine_mean)
            and quarantine_mean
            >= gates["min_quarantine_mean_ratio"] * development_mean
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "promoted": not failed,
        "failed_gates": failed,
        "checks": checks,
        "development_selected_mean": development_mean,
        "quarantine_selected_mean": quarantine_mean,
        "pbo": pbo,
    }


def run_candidate(name: str, model: DummyModel) -> dict:
    frame = make_triple_barrier_frame()
    frame["model_score"] = model.predict_scores(frame)
    frame["selected"] = frame["model_score"].ge(0.60)
    roles = ColumnRoleMap(
        timestamp="event_time",
        label="triple_barrier_label",
        selected="selected",
        outcome="economic_return",
        high="high",
        low="low",
        close="close",
        side="side",
    )

    pipeline = FullPipeline.validation_pipeline(
        source=frame,
        roles=roles,
        artifact_dir=Path("outputs") / name,
        label_audit=LabelAuditConfig(
            min_labeled_rows=100,
            allowed_label_values=(-1, 0, 1),
        ),
        holdout=HoldoutConfig(quarantine_fraction=0.20),
        cpcv=CPCVConfig(
            n_groups=6,
            holdout_groups=2,
            validate_groups=1,
            purge_bars=1,
            embargo_bars=1,
            max_splits=12,
        ),
        bootstrap=FoldBlockBootstrapConfig(
            replicates=30,
            block_bars=8,
            random_seed=7,
        ),
        null_tests=NullTestConfig(
            random_simulations=500,
            stored_random_paths=60,
            random_seed=7,
            min_random_percentile=promotion_gates["min_random_percentile"],
            max_p_value=promotion_gates["max_null_p_value"],
        ),
        triple_barrier=TRIPLE_BARRIER,
        plots=PlotConfig(
            enabled=True,
            max_paths=60,
            bootstrap_block_bars=8,
            include_fold_bootstrap=True,
            random_seed=7,
        ),
        risk_metrics=RiskMetricsConfig(
            min_selected=int(promotion_gates["min_selected_trades"]),
        ),
        fail_fast=False,
    )
    run_result = pipeline.run()
    run_result.context.state["candidate_metrics"] = build_pbo_metrics(
        run_result.context,
        thresholds=(0.55, 0.60, 0.65),
    )
    pbo_result = PBOAnalyzer(
        PBOConfig(
            min_folds=6,
            max_pbo=promotion_gates["max_pbo"],
            max_upper_confidence_bound=1.0,
        )
    ).run(run_result.context)
    return evaluate_promotion(run_result, pbo_result, promotion_gates)


reports = {
    "passing_candidate": run_candidate(
        "passing_candidate",
        DummyModel(invert=False),
    ),
    "failing_candidate": run_candidate(
        "failing_candidate",
        DummyModel(invert=True),
    ),
}

for candidate_name, report in reports.items():
    print(candidate_name, report)

assert reports["passing_candidate"]["promoted"] is True
assert reports["failing_candidate"]["promoted"] is False
```
<!-- full-pipeline-example:end -->

The deterministic example classifies the candidates as follows:

```text
passing_candidate: promoted=True, failed_gates=[]
failing_candidate: promoted=False, failed_gates=[pipeline_passed, null_p_value,
random_percentile, mean_return, max_drawdown, quarantine_degradation]
```

The rejected candidate can still report low PBO because PBO measures selection
instability, not profitability. That is why the economic, null, drawdown, and
quarantine gates remain separate.

The run writes these figures under `artifact_dir/plots/`:

- `label_diagnostics.png`: real cumulative outcome path, randomized outcome
  paths, and class balance.
- `cpcv_distributions.png`: CPCV validate/test distributions and optional
  per-fold train block-bootstrap bell curves.
- `null_test_equity_paths.png`: selected-trade equity against the exact
  same-count random samples used by the null test.
- `triple_barrier_bootstrap_paths.png`: economic barrier-return bootstrap paths
  with the pointwise median in black.

Set `strict=True` in `PlotConfig` to fail a plotting component when its required
input is unavailable. With the default `strict=False`, unavailable optional plot
inputs are reported as skipped and do not change statistical pass/fail results.

## Data Assumptions

- Input can be a pandas DataFrame, CSV file, or Parquet file.
- Timestamps must be parseable and are normalized to UTC and sorted by default.
- Column names are user-defined through `ColumnRoleMap`; no fixed schema names
  are required.
- The outcome column must contain additive per-event or per-trade values for
  cumulative equity and aggregate distribution plots.
- CPCV plots use validate and test indices exactly as constructed by the CPCV
  component. Train bootstrap results are shown separately.
- Randomized plots are deterministic for a fixed seed.
- Triple-barrier economic returns use configured target and stop returns;
  vertical timeouts use side-adjusted close-to-close returns.
- Plot generation follows pipeline `fail_fast` behavior. Use `fail_fast=False`
  when diagnostics from later failed stages must still be attempted.

Run tests locally:

```powershell
.\.venv\Scripts\python.exe -m pytest
```
