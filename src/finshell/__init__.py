"""finShell validation engine."""

from finshell.core import ComponentResult, FullPipeline, PipelineComponent, PipelineContext, PipelineRunResult
from finshell.ingestion import ColumnRoleMap, DataIngestConfig, DataIngestor

__version__ = "0.1.0"

__all__ = [
    "ColumnRoleMap",
    "ComponentResult",
    "DataIngestConfig",
    "DataIngestor",
    "FullPipeline",
    "PipelineComponent",
    "PipelineContext",
    "PipelineRunResult",
    "__version__",
]
