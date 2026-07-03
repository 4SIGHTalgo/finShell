"""finShell validation engine."""

from finshell.core import ComponentResult, FullPipeline, PipelineComponent, PipelineContext, PipelineRunResult
from finshell.cpcv import CPCVConfig, CPCVFold, CPCVPurgeEmbargo
from finshell.ingestion import ColumnRoleMap, DataIngestConfig, DataIngestor
from finshell.holdout import HoldoutConfig, HoldoutSplitter
from finshell.label_audit import LabelAuditConfig, LabelAuditor

__version__ = "0.1.0"

__all__ = [
    "ColumnRoleMap",
    "ComponentResult",
    "CPCVConfig",
    "CPCVFold",
    "CPCVPurgeEmbargo",
    "DataIngestConfig",
    "DataIngestor",
    "FullPipeline",
    "HoldoutConfig",
    "HoldoutSplitter",
    "LabelAuditConfig",
    "LabelAuditor",
    "PipelineComponent",
    "PipelineContext",
    "PipelineRunResult",
    "__version__",
]
