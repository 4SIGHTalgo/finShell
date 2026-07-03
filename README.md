# finShell

Python validation engine for model-free financial label and backtest audits.

## Installation

Install the core package from a local checkout:

```powershell
python -m pip install .
```

Install plotting support:

```powershell
python -m pip install ".[plots]"
```

`finShell` is built around a role-mapped, OOP pipeline:

1. ingest user data without hardcoded column names
2. audit label and feature contracts
3. seal a final quarantine holdout
4. build CPCV purge/embargo folds
5. bootstrap fold train paths
6. run selected-trade null tests
7. compute risk-adjusted path metrics

Plotting is opt-in. The statistical pipeline does not import Matplotlib or write
figures unless a `PlotConfig` with `enabled=True` is supplied.

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
