from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pandas as pd

from finshell.core import PipelineContext
from finshell.ingestion import ColumnRoleMap


def test_plotting_module_exposes_opt_in_configuration() -> None:
    spec = importlib.util.find_spec("finshell.plotting")

    assert spec is not None
    module = importlib.import_module("finshell.plotting")
    config = module.PlotConfig()
    assert config.enabled is False
    assert config.output_subdir == "plots"
    assert config.max_paths == 100


def _label_context(artifact_dir: Path) -> PipelineContext:
    context = PipelineContext(artifact_dir=artifact_dir)
    context.state["data"] = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC"),
            "label": [-1, 0, 1, 1, -1, 1],
            "outcome": [-0.02, 0.0, 0.03, 0.01, -0.01, 0.04],
        }
    )
    context.state["roles"] = ColumnRoleMap(
        timestamp="event_time",
        label="label",
        outcome="outcome",
    )
    return context


def test_label_plot_is_disabled_without_filesystem_side_effects(tmp_path: Path) -> None:
    module = importlib.import_module("finshell.plotting")
    component_type = getattr(module, "LabelDiagnosticsPlot", None)

    assert component_type is not None
    context = _label_context(tmp_path)
    result = component_type(module.PlotConfig()).run(context)
    assert result.passed is True
    assert result.summary == {"skipped": True, "reason": "plotting_disabled"}
    assert list(tmp_path.rglob("*")) == []


def test_label_plot_writes_deterministic_equity_and_class_balance(tmp_path: Path) -> None:
    module = importlib.import_module("finshell.plotting")
    component_type = getattr(module, "LabelDiagnosticsPlot", None)

    assert component_type is not None
    config = module.PlotConfig(enabled=True, max_paths=4, random_seed=7)
    first = _label_context(tmp_path / "first")
    second = _label_context(tmp_path / "second")

    first_result = component_type(config).run(first)
    second_result = component_type(config).run(second)

    first_path = Path(first_result.artifacts["figure"])
    assert first_result.passed is True
    assert first_path.name == "label_diagnostics.png"
    assert first_path.stat().st_size > 0
    assert first_result.summary["class_counts"] == {"-1": 2, "0": 1, "1": 3}
    assert first.state["label_plot_diagnostics"]["real_equity"] == [
        -0.02,
        -0.02,
        0.01,
        0.02,
        0.01,
        0.05,
    ]
    assert first.state["label_plot_diagnostics"]["random_equity_paths"] == second.state[
        "label_plot_diagnostics"
    ]["random_equity_paths"]
