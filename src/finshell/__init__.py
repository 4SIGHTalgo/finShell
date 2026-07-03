"""finShell validation engine."""

from finshell.bootstrap import (
    BOOTSTRAP_CONTRACT_VERSION,
    BootstrapPath,
    FoldBlockBootstrap,
    FoldBlockBootstrapConfig,
    FoldBootstrapPlan,
    infer_block_bars,
)
from finshell.core import ComponentResult, FullPipeline, PipelineComponent, PipelineContext, PipelineRunResult
from finshell.cpcv import CPCVConfig, CPCVFold, CPCVPurgeEmbargo
from finshell.ingestion import ColumnRoleMap, DataIngestConfig, DataIngestor
from finshell.holdout import HoldoutConfig, HoldoutSplitter
from finshell.label_audit import LabelAuditConfig, LabelAuditor
from finshell.null_tests import NullTestConfig, NullTestSuite
from finshell.pbo import PBOAnalyzer, PBOConfig, binomial_upper_confidence_bound

__version__ = "0.1.0"

__all__ = [
    "ColumnRoleMap",
    "ComponentResult",
    "BOOTSTRAP_CONTRACT_VERSION",
    "BootstrapPath",
    "CPCVConfig",
    "CPCVFold",
    "CPCVPurgeEmbargo",
    "DataIngestConfig",
    "DataIngestor",
    "FoldBlockBootstrap",
    "FoldBlockBootstrapConfig",
    "FoldBootstrapPlan",
    "FullPipeline",
    "HoldoutConfig",
    "HoldoutSplitter",
    "LabelAuditConfig",
    "LabelAuditor",
    "NullTestConfig",
    "NullTestSuite",
    "PBOAnalyzer",
    "PBOConfig",
    "PipelineComponent",
    "PipelineContext",
    "PipelineRunResult",
    "binomial_upper_confidence_bound",
    "infer_block_bars",
    "__version__",
]
