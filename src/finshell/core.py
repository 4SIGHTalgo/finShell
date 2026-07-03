from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class PipelineContext:
    artifact_dir: Path | str = Path("outputs/finshell")
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.artifact_dir = Path(self.artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, relative_path: str | Path) -> Path:
        path = self.artifact_dir / Path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(frozen=True, slots=True)
class ComponentResult:
    component: str
    passed: bool
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path | str] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = {key: str(value) for key, value in self.artifacts.items()}
        return payload


class PipelineComponent(ABC):
    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def run(self, context: PipelineContext) -> ComponentResult:
        raise NotImplementedError

    def to_manifest(self) -> dict[str, Any]:
        return {"name": self.name, "class": self.__class__.__name__}

    def validate_contract(self) -> list[str]:
        return []


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    passed: bool
    context: PipelineContext
    components: list[ComponentResult]
    manifest: dict[str, Any]


class FullPipeline:
    def __init__(
        self,
        components: Iterable[PipelineComponent] | None = None,
        *,
        artifact_dir: Path | str = Path("outputs/finshell"),
        fail_fast: bool = True,
    ) -> None:
        self.components = list(components or [])
        self.artifact_dir = Path(artifact_dir)
        self.fail_fast = bool(fail_fast)

    @classmethod
    def from_components(
        cls,
        components: Iterable[PipelineComponent],
        *,
        artifact_dir: Path | str = Path("outputs/finshell"),
        fail_fast: bool = True,
    ) -> FullPipeline:
        return cls(components=components, artifact_dir=artifact_dir, fail_fast=fail_fast)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> FullPipeline:
        payload = dict(config or {})
        return cls(
            components=payload.get("components") or [],
            artifact_dir=payload.get("artifact_dir", Path("outputs/finshell")),
            fail_fast=payload.get("fail_fast", True),
        )

    @classmethod
    def validation_pipeline(
        cls,
        *,
        source: Any,
        roles: Any,
        artifact_dir: Path | str = Path("outputs/finshell"),
        label_audit: Any | None = None,
        holdout: Any | None = None,
        cpcv: Any | None = None,
        bootstrap: Any | None = None,
        null_tests: Any | None = None,
        risk_metrics: Any | None = None,
        fail_fast: bool = True,
    ) -> FullPipeline:
        from finshell.bootstrap import FoldBlockBootstrap, FoldBlockBootstrapConfig
        from finshell.cpcv import CPCVConfig, CPCVPurgeEmbargo
        from finshell.holdout import HoldoutConfig, HoldoutSplitter
        from finshell.ingestion import DataIngestConfig, DataIngestor
        from finshell.label_audit import LabelAuditConfig, LabelAuditor
        from finshell.null_tests import NullTestConfig, NullTestSuite
        from finshell.risk_metrics import RiskMetrics, RiskMetricsConfig

        components: list[PipelineComponent] = [
            DataIngestor(DataIngestConfig(source=source, roles=roles)),
            LabelAuditor(label_audit or LabelAuditConfig()),
            HoldoutSplitter(holdout or HoldoutConfig()),
            CPCVPurgeEmbargo(cpcv or CPCVConfig()),
            FoldBlockBootstrap(bootstrap or FoldBlockBootstrapConfig()),
            NullTestSuite(null_tests or NullTestConfig()),
            RiskMetrics(risk_metrics or RiskMetricsConfig()),
        ]
        return cls.from_components(components, artifact_dir=artifact_dir, fail_fast=fail_fast)

    def run(self, context: PipelineContext | None = None) -> PipelineRunResult:
        active_context = context or PipelineContext(artifact_dir=self.artifact_dir)
        results: list[ComponentResult] = []
        for component in self.components:
            contract_errors = component.validate_contract()
            if contract_errors:
                result = ComponentResult(
                    component=component.name,
                    passed=False,
                    summary={"contract_errors": contract_errors},
                )
            else:
                result = component.run(active_context)
            results.append(result)
            if self.fail_fast and not result.passed:
                break
        passed = all(result.passed for result in results)
        manifest = {
            "pipeline": self.__class__.__name__,
            "passed": passed,
            "fail_fast": self.fail_fast,
            "components": [result.to_manifest() for result in results],
        }
        return PipelineRunResult(
            passed=passed,
            context=active_context,
            components=results,
            manifest=manifest,
        )
