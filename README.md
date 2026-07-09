# finShell
<img width="1254" height="1254" alt="ChatGPT Image Jul 9, 2026, 04_09_25 AM" src="https://github.com/user-attachments/assets/7a727763-a200-4b46-a2b1-e8e2e157b01c" />


Python validation engine for financial labels and cross-validated selectors.

finShell is a validation framework, not a signal generator or trading system.
Passing finShell diagnostics does not imply live profitability.
It is designed to reduce false confidence from overfit labels, selectors, and backtests.

## Why finShell exists

Financial ML experiments are easy to overfit. A label can look predictive because
the researcher repeatedly changed horizons, barriers, features, filters, or
thresholds after seeing historical results.

finShell gives researchers a structured way to audit:

1. whether a label beats random same-count paths,
2. whether a selector survives purged CPCV and block bootstrap,
3. whether selected quarantine trades beat random selection,
4. whether selected outcomes survive economic path simulation.

## Installation

```bash
python -m pip install finshell
```

The standard installation includes plotting, CPCV, block bootstrap, null tests,
triple-barrier labels, logistic selectors, and sealed out-of-sample validation.
Input columns are mapped with `ColumnRoleMap`, so source schemas do not need to
use finShell's internal names.

For local development:

```bash
git clone https://github.com/4SIGHTalgo/finShell
cd finShell
python -m pip install -e ".[dev]"
python -m pytest
```

## Quick start with real data

```python
import finshell as fs

study = fs.ValidationStudy(
    "my_ohlc_events.parquet",
    roles=fs.ColumnRoleMap(
        timestamp="timestamp",
        high="high",
        low="low",
        close="close",
        side="side",
    ),
)

label = study.audit_label(
    fs.TripleBarrierConfig(profit_take=0.015, stop_loss=0.010, vertical_bars=16),
)

cv = study.fit_selector(
    fs.LogisticSelector(features=["volatility", "trend", "vix_change"], threshold=0.60),
)

oos = study.audit_oos()
economics = study.validate_economics()
```

## Notebook Walkthrough

This deterministic example creates an original triple-barrier classification
label from seeded mock OHLC data. The mock market has persistent latent signal,
irregular noise, and a small adverse drift so its unfiltered equity is not a
stylized sine wave. It is an API example, not evidence of a tradable edge.

<!-- notebook-example:start -->

### 1. Generate and audit the label

The first operation after creating the label is a same-count random-path audit.
The figure compares the favorable-label equity path with the null paths and
their dashed pointwise p95 boundary, then reports class balance.

```python
import numpy as np
import pandas as pd
import finshell as fs

rows = 480
rng = np.random.default_rng(2024)
signal = np.zeros(rows)
innovations = rng.normal(0.0, 0.75, rows)
for index in range(1, rows):
    signal[index] = 0.82 * signal[index - 1] + innovations[index]
signal = (signal - signal.mean()) / signal.std()
market_noise = rng.normal(0.0, 0.004, rows - 1)
next_returns = 0.0040 * np.tanh(signal[:-1]) + market_noise - 0.0010
close = np.empty(rows)
close[0] = 100.0
for index in range(rows - 1):
    close[index + 1] = close[index] * (1.0 + next_returns[index])
intrabar = np.abs(rng.normal(0.0012, 0.0005, rows))

frame = pd.DataFrame({
    "event_time": pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC"),
    "signal_feature": signal,
    "high": close * (1.0 + intrabar),
    "low": close * (1.0 - intrabar),
    "close": close,
    "side": np.ones(rows, dtype=int),
})
study = fs.ValidationStudy(
    frame,
    roles=fs.ColumnRoleMap(
        timestamp="event_time", high="high", low="low", close="close", side="side"
    ),
    artifact_dir="assets/readme",
)
label = study.audit_label(
    fs.TripleBarrierConfig(profit_take=0.015, stop_loss=0.012, vertical_bars=8),
    null_tests=fs.NullTestConfig(
        random_simulations=300, stored_random_paths=40, random_seed=7
    ),
)
print(
    f"passed={label.passed} favorable={label.summary['favorable_count']} "
    f"real_total={label.summary['real_total']:.4f} "
    f"random_p95={label.summary['random_final_p95']:.4f} "
    f"p_value={label.summary['p_value']:.4f}"
)
```

```text
passed=True favorable=73 real_total=1.0950 random_p95=-0.1088 p_value=0.0000
```

![Label audit](https://raw.githubusercontent.com/4SIGHTalgo/finShell/main/assets/readme/plots/01_label_audit.png)

### 2. Fit inside CPCV folds

`LogisticSelector` is fitted separately on every block-bootstrap training path.
Validation and test rows are never included in those fits, and the final 20%
quarantine remains sealed. The left panel shows the distribution across CPCV
fold means. The right panel preserves the hierarchy: each CPCV permutation is a
row containing its validation and test block-bootstrap distributions.

```python
cv = study.fit_selector(
    fs.LogisticSelector(
        features=["signal_feature"], threshold=0.30, random_state=13
    ),
    cpcv=fs.CPCVConfig(
        n_groups=6, holdout_groups=2, validate_groups=1, max_splits=4
    ),
    bootstrap=fs.FoldBlockBootstrapConfig(
        replicates=4, block_bars=8, random_seed=13
    ),
)
print(
    f"passed={cv.passed} valid_bootstrap_fits={cv.summary['valid_bootstrap_fits']} "
    f"validate_return={cv.summary['mean_validate_total_return']:.4f} "
    f"test_return={cv.summary['mean_test_total_return']:.4f}"
)
```

```text
passed=True valid_bootstrap_fits=16 validate_return=0.0206 test_return=0.0589
```

![CPCV selector distributions](https://raw.githubusercontent.com/4SIGHTalgo/finShell/main/assets/readme/plots/02_cpcv_selector.png)

### 3. Audit selected quarantine trades

The already-fitted selector scores the sealed quarantine without refitting. Its
equity path must beat both same-count random selection p95 and the unfiltered
no-selector equity path.

```python
oos = study.audit_oos(
    null_tests=fs.NullTestConfig(
        random_simulations=300, stored_random_paths=40, random_seed=17
    )
)
print(
    f"passed={oos.passed} selected={oos.summary['selected_count']} "
    f"selected_total={oos.summary['selected_total']:.4f} "
    f"random_p95={oos.summary['random_final_p95']:.4f} "
    f"no_selector={oos.summary['no_selector_total']:.4f} "
    f"p_value={oos.summary['p_value']:.4f}"
)
```

```text
passed=True selected=36 selected_total=0.3088 random_p95=0.1589 no_selector=0.1981 p_value=0.0000
```

![Out-of-sample selected-trade audit](https://raw.githubusercontent.com/4SIGHTalgo/finShell/main/assets/readme/plots/03_oos_audit.png)

### 4. Validate economic paths

The last stage block-bootstraps the selected OOS backtest outcomes into account-
balance paths. Each path ends at the configured upper balance, lower balance,
or trade-count horizon. The single chart reports the probability of every
resolution state and the median number of trades to resolution.

```python
economics = study.validate_economics(
    paths=300,
    block_bars=4,
    random_seed=23,
    initial_balance=100_000,
    upper_balance=105_000,
    lower_balance=95_000,
    max_trades=12,
)
print(
    f"passed={economics.passed} paths={economics.summary['bootstrap_paths']} "
    f"upper_hit={economics.summary['upper_hit_probability']:.1%} "
    f"lower_hit={economics.summary['lower_hit_probability']:.1%} "
    f"vertical={economics.summary['vertical_probability']:.1%} "
    f"median_resolution={economics.summary['median_resolution_trades']:.1f} trades"
)
```

```text
passed=True paths=300 upper_hit=89.7% lower_hit=1.0% vertical=9.3% median_resolution=5.0 trades
```

![Triple-barrier economic validation](https://raw.githubusercontent.com/4SIGHTalgo/finShell/main/assets/readme/plots/04_economic_monte_carlo.png)

<!-- notebook-example:end -->

## Data Assumptions

- Input may be a pandas DataFrame, CSV file, or Parquet file.
- Timestamps must be parseable; ingestion normalizes them to UTC and sorts them.
- `ColumnRoleMap` maps user-defined column names into the validation contract.
- OHLC and side columns are required when finShell generates triple-barrier labels.
- Outcomes are additive per-event returns used for cumulative equity and path tests.
- Randomized and bootstrap results are deterministic for a fixed seed.
- Quarantine data is the final chronological 20% by default and is not used in CPCV.
- Economic barrier paths resample contiguous selected-trade outcomes from the
  OOS backtest and compound them from the configured initial account balance.

Run the test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```
