from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finshell.bootstrap import FoldBlockBootstrap, FoldBlockBootstrapConfig
from finshell.core import PipelineContext
from finshell.cpcv import CPCVConfig, CPCVPurgeEmbargo
from finshell.ingestion import ColumnRoleMap
from finshell.null_tests import NullTestConfig, NullTestSuite


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


def _cpcv_context(artifact_dir: Path) -> PipelineContext:
    context = PipelineContext(artifact_dir=artifact_dir)
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC"),
            "label": [index % 3 - 1 for index in range(24)],
            "outcome": [(index - 8) / 100.0 for index in range(24)],
        }
    )
    context.state["development_data"] = frame
    context.state["roles"] = ColumnRoleMap(
        timestamp="event_time",
        label="label",
        outcome="outcome",
    )
    CPCVPurgeEmbargo(
        CPCVConfig(n_groups=4, holdout_groups=2, validate_groups=1, max_splits=2)
    ).run(context)
    FoldBlockBootstrap(
        FoldBlockBootstrapConfig(replicates=3, block_bars=2, random_seed=11)
    ).run(context)
    return context


def test_cpcv_plot_uses_declared_partitions_and_bootstrap_paths(tmp_path: Path) -> None:
    module = importlib.import_module("finshell.plotting")
    component_type = getattr(module, "CPCVDiagnosticsPlot", None)

    assert component_type is not None
    context = _cpcv_context(tmp_path)
    result = component_type(
        module.PlotConfig(enabled=True, include_fold_bootstrap=True)
    ).run(context)

    frame = context.state["development_data"]
    outcomes = frame["outcome"]
    folds = context.state["cpcv_folds"]
    plans = context.state["bootstrap_plans"]
    diagnostics = context.state["cpcv_plot_diagnostics"]
    assert diagnostics["validate_totals"] == pytest.approx(
        [float(outcomes.iloc[fold.validate_indices].sum()) for fold in folds]
    )
    assert diagnostics["test_totals"] == pytest.approx(
        [float(outcomes.iloc[fold.test_indices].sum()) for fold in folds]
    )
    assert diagnostics["validate_fit"] == pytest.approx(
        {
            "mean": float(np.mean(diagnostics["validate_totals"])),
            "std": float(np.std(diagnostics["validate_totals"], ddof=1)),
        }
    )
    assert diagnostics["test_fit"] == pytest.approx(
        {
            "mean": float(np.mean(diagnostics["test_totals"])),
            "std": float(np.std(diagnostics["test_totals"], ddof=1)),
        }
    )
    for fold_diagnostic, plan in zip(diagnostics["folds"], plans):
        assert fold_diagnostic["bootstrap_totals"] == pytest.approx(
            [float(outcomes.iloc[path.indices].sum()) for path in plan.paths]
        )
        assert fold_diagnostic["bootstrap_fit"] == pytest.approx(
            {
                "mean": float(np.mean(fold_diagnostic["bootstrap_totals"])),
                "std": float(np.std(fold_diagnostic["bootstrap_totals"], ddof=1)),
            }
        )
    figure_path = Path(result.artifacts["figure"])
    assert figure_path.name == "cpcv_distributions.png"
    assert figure_path.stat().st_size > 0


def test_null_plot_consumes_exact_statistical_samples(tmp_path: Path) -> None:
    module = importlib.import_module("finshell.plotting")
    component_type = getattr(module, "NullTestDiagnosticsPlot", None)

    assert component_type is not None
    context = PipelineContext(artifact_dir=tmp_path)
    context.state["development_data"] = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC"),
            "label": [-1, -1, 0, 0, 1, 1, 1, 1],
            "selected": [False, False, False, False, True, True, True, True],
            "outcome": [-0.03, -0.02, -0.01, 0.0, 0.05, 0.06, 0.07, 0.08],
        }
    )
    context.state["roles"] = ColumnRoleMap(
        timestamp="event_time",
        label="label",
        selected="selected",
        outcome="outcome",
    )
    NullTestSuite(NullTestConfig(random_simulations=20, random_seed=13)).run(context)

    result = component_type(module.PlotConfig(enabled=True, max_paths=5)).run(context)

    statistical = context.state["null_test_diagnostics"]
    plotted = context.state["null_plot_diagnostics"]
    frame = context.state["development_data"]
    expected_paths = [
        [
            float(value)
            for value in frame.iloc[indices]["outcome"].cumsum().round(12).tolist()
        ]
        for indices in statistical["random_row_indices"][:5]
    ]
    assert plotted["sampled_row_indices"] == statistical["random_row_indices"][:5]
    assert plotted["random_equity_paths"] == expected_paths
    assert plotted["real_equity"] == [0.05, 0.11, 0.18, 0.26]
    figure_path = Path(result.artifacts["figure"])
    assert figure_path.name == "null_test_equity_paths.png"
    assert figure_path.stat().st_size > 0
