from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from finshell.core import PipelineContext
from finshell.ingestion import ColumnRoleMap, DataIngestConfig, DataIngestor


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_time": ["2026-01-01T02:00:00Z", "2026-01-01T01:00:00Z"],
            "tb_label": [1, 0],
            "selected": [True, False],
            "net_return": [0.02, -0.01],
            "close_px": [101.0, 100.0],
        }
    )


def test_ingestor_accepts_dataframe_and_sorts_by_role_timestamp(tmp_path: Path) -> None:
    roles = ColumnRoleMap(
        timestamp="event_time",
        label="tb_label",
        selected="selected",
        outcome="net_return",
        close="close_px",
    )
    context = PipelineContext(artifact_dir=tmp_path)

    result = DataIngestor(DataIngestConfig(source=_raw_frame(), roles=roles)).run(context)

    ingested = context.state["data"]
    assert result.passed is True
    assert ingested["event_time"].tolist() == [
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T02:00:00Z"),
    ]
    assert context.state["roles"] == roles
    assert result.summary["rows"] == 2
    assert result.summary["timestamp_min_utc"] == "2026-01-01T01:00:00+00:00"


def test_ingestor_loads_csv_and_parquet_by_file_extension(tmp_path: Path) -> None:
    frame = _raw_frame()
    csv_path = tmp_path / "labels.csv"
    parquet_path = tmp_path / "labels.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)
    roles = ColumnRoleMap(timestamp="event_time", label="tb_label")

    csv_result = DataIngestor(DataIngestConfig(source=csv_path, roles=roles)).run(PipelineContext(tmp_path / "csv"))
    parquet_result = DataIngestor(DataIngestConfig(source=parquet_path, roles=roles)).run(
        PipelineContext(tmp_path / "parquet")
    )

    assert csv_result.passed is True
    assert parquet_result.passed is True
    assert csv_result.summary["source_format"] == "csv"
    assert parquet_result.summary["source_format"] == "parquet"


def test_ingestor_fails_closed_when_required_role_column_is_missing(tmp_path: Path) -> None:
    roles = ColumnRoleMap(timestamp="event_time", label="missing_label")

    with pytest.raises(ValueError, match="missing required role columns: label=missing_label"):
        DataIngestor(DataIngestConfig(source=_raw_frame(), roles=roles)).run(PipelineContext(tmp_path))
