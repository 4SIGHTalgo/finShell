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
from finshell.plotting import CPCVDiagnosticsPlot, LabelDiagnosticsPlot, PlotConfig
from finshell.risk_metrics import RiskMetrics, RiskMetricsConfig, compute_risk_metrics
from finshell.triple_barrier import TripleBarrierComparator, TripleBarrierConfig

__version__ = "0.1.0"

__all__ = [
    "ColumnRoleMap",
    "ComponentResult",
    "BOOTSTRAP_CONTRACT_VERSION",
    "BootstrapPath",
    "CPCVConfig",
    "CPCVFold",
    "CPCVPurgeEmbargo",
    "CPCVDiagnosticsPlot",
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
    "LabelDiagnosticsPlot",
    "NullTestConfig",
    "NullTestSuite",
    "PBOAnalyzer",
    "PBOConfig",
    "PipelineComponent",
    "PipelineContext",
    "PipelineRunResult",
    "PlotConfig",
    "RiskMetrics",
    "RiskMetricsConfig",
    "TripleBarrierComparator",
    "TripleBarrierConfig",
    "binomial_upper_confidence_bound",
    "compute_risk_metrics",
    "infer_block_bars",
    "__version__",
]
