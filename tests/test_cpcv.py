from __future__ import annotations

from pathlib import Path

import pandas as pd

from finshell.core import PipelineContext
from finshell.cpcv import CPCVConfig, CPCVPurgeEmbargo
from finshell.ingestion import ColumnRoleMap


def _context(tmp_path: Path, rows: int = 8) -> PipelineContext:
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC"),
            "tb_label": [1 if idx % 2 else -1 for idx in range(rows)],
        }
    )
    context = PipelineContext(tmp_path)
    context.state["development_data"] = frame
    context.state["roles"] = ColumnRoleMap(timestamp="event_time", label="tb_label")
    return context


def test_cpcv_builds_combinatorial_validate_test_folds(tmp_path: Path) -> None:
    context = _context(tmp_path)

    result = CPCVPurgeEmbargo(CPCVConfig(n_groups=4, holdout_groups=2, validate_groups=1)).run(context)

    folds = context.state["cpcv_folds"]
    assert result.passed is True
    assert result.summary["fold_count"] == 6
    assert len(folds) == 6
    assert folds[0].validate_group_indices == [1]
    assert folds[0].test_group_indices == [2]


def test_cpcv_masks_keep_train_validate_and_test_disjoint(tmp_path: Path) -> None:
    context = _context(tmp_path)

    CPCVPurgeEmbargo(CPCVConfig(n_groups=4, holdout_groups=2, validate_groups=1)).run(context)
    fold = context.state["cpcv_folds"][0]

    train = set(fold.train_indices)
    validate = set(fold.validate_indices)
    test = set(fold.test_indices)
    assert train.isdisjoint(validate)
    assert train.isdisjoint(test)
    assert validate.isdisjoint(test)


def test_cpcv_purge_embargo_removes_neighboring_train_rows(tmp_path: Path) -> None:
    context = _context(tmp_path)

    CPCVPurgeEmbargo(
        CPCVConfig(
            n_groups=4,
            holdout_groups=2,
            validate_groups=1,
            purge_bars=1,
            embargo_bars=1,
            max_splits=1,
        )
    ).run(context)
    fold = context.state["cpcv_folds"][0]

    assert fold.validate_indices == [0, 1]
    assert fold.test_indices == [2, 3]
    assert 4 not in fold.train_indices
    assert fold.train_indices == [5, 6, 7]


def test_cpcv_records_cache_manifest(tmp_path: Path) -> None:
    context = _context(tmp_path)

    CPCVPurgeEmbargo(CPCVConfig(n_groups=4, holdout_groups=2, validate_groups=1, max_splits=2)).run(context)

    manifest = context.state["cpcv_manifest"]
    assert manifest["cv_type"] == "combinatorial_purged_k_fold"
    assert manifest["fold_count"] == 2
    assert manifest["purge_bars"] == 0
