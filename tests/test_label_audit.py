from __future__ import annotations

from pathlib import Path

import pandas as pd

from finshell.core import PipelineContext
from finshell.ingestion import ColumnRoleMap
from finshell.label_audit import LabelAuditConfig, LabelAuditor


def _context_with_frame(tmp_path: Path, frame: pd.DataFrame, roles: ColumnRoleMap) -> PipelineContext:
    context = PipelineContext(tmp_path)
    context.state["data"] = frame.copy()
    context.state["roles"] = roles
    return context


def test_label_auditor_accepts_clean_triple_barrier_classification_labels(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"),
            "label_end": pd.date_range("2026-01-01T00:30:00Z", periods=4, freq="1h"),
            "tb_label": [-1, 0, 1, 1],
            "net_return": [-0.01, 0.0, 0.02, 0.03],
        }
    )
    roles = ColumnRoleMap(timestamp="event_time", label="tb_label", outcome="net_return", label_end_timestamp="label_end")

    result = LabelAuditor(LabelAuditConfig(min_labeled_rows=3)).run(_context_with_frame(tmp_path, frame, roles))

    assert result.passed is True
    assert result.summary["label_counts"] == {"-1": 1, "0": 1, "1": 2}
    assert result.summary["fail_reasons"] == []


def test_label_auditor_fails_closed_for_unknown_label_values(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
            "tb_label": [-1, 2, 1],
        }
    )
    roles = ColumnRoleMap(timestamp="event_time", label="tb_label")

    result = LabelAuditor(LabelAuditConfig(min_labeled_rows=3)).run(_context_with_frame(tmp_path, frame, roles))

    assert result.passed is False
    assert "label_value_outside_allowed_set" in result.summary["fail_reasons"]


def test_label_auditor_detects_future_observed_timestamp_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
            "macro_observed_timestamp_utc": pd.date_range("2026-01-01T00:30:00Z", periods=3, freq="1h"),
            "tb_label": [-1, 0, 1],
        }
    )
    roles = ColumnRoleMap(timestamp="event_time", label="tb_label")

    result = LabelAuditor(LabelAuditConfig(min_labeled_rows=3)).run(_context_with_frame(tmp_path, frame, roles))

    assert result.passed is False
    assert result.summary["observed_timestamp_violations"] == 3
    assert "observed_timestamp_violation" in result.summary["fail_reasons"]


def test_label_auditor_detects_excessive_interval_overlap(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"),
            "label_end": pd.to_datetime(
                [
                    "2026-01-01T03:00:00Z",
                    "2026-01-01T04:00:00Z",
                    "2026-01-01T05:00:00Z",
                    "2026-01-01T06:00:00Z",
                ],
                utc=True,
            ),
            "tb_label": [-1, 0, 1, 1],
        }
    )
    roles = ColumnRoleMap(timestamp="event_time", label="tb_label", label_end_timestamp="label_end")

    result = LabelAuditor(LabelAuditConfig(min_labeled_rows=3, max_overlap_rate=0.5)).run(
        _context_with_frame(tmp_path, frame, roles)
    )

    assert result.passed is False
    assert result.summary["interval_overlap_rate"] > 0.5
    assert "excessive_label_overlap" in result.summary["fail_reasons"]


def test_label_auditor_reports_feature_contract_roles(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
            "tb_label": [-1, 0, 1],
            "future_return": [0.1, -0.1, 0.0],
            "distance_feature": [1.0, 2.0, 3.0],
        }
    )
    roles = ColumnRoleMap(timestamp="event_time", label="tb_label")

    result = LabelAuditor(LabelAuditConfig(min_labeled_rows=3)).run(_context_with_frame(tmp_path, frame, roles))

    contract = {row["column"]: row["role"] for row in result.summary["feature_contract"]}
    assert contract["future_return"] == "leakage_risk_do_not_use_as_feature"
    assert contract["distance_feature"] == "candidate_feature"
