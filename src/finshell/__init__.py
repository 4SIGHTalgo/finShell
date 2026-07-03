"""finShell validation engine."""

from finshell.core import ComponentResult, FullPipeline, PipelineComponent, PipelineContext, PipelineRunResult
from finshell.ingestion import ColumnRoleMap, DataIngestConfig, DataIngestor
from finshell.label_audit import LabelAuditConfig, LabelAuditor

__version__ = "0.1.0"

__all__ = [
    "ColumnRoleMap",
    "ComponentResult",
    "DataIngestConfig",
    "DataIngestor",
    "FullPipeline",
    "LabelAuditConfig",
    "LabelAuditor",
    "PipelineComponent",
    "PipelineContext",
    "PipelineRunResult",
    "__version__",
]
