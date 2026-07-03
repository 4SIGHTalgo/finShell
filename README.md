# finShell

Python validation engine for model-free financial label and backtest audits.

`finShell` is built around a role-mapped, OOP pipeline:

1. ingest user data without hardcoded column names
2. audit label and feature contracts
3. seal a final quarantine holdout
4. build CPCV purge/embargo folds
5. bootstrap fold train paths
6. run selected-trade null tests
7. compute risk-adjusted path metrics

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

Run tests locally:

```powershell
.\.venv\Scripts\python.exe -m pytest
```
