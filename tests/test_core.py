from __future__ import annotations

from pathlib import Path

from finshell.core import ComponentResult, FullPipeline, PipelineComponent, PipelineContext


class RecordingComponent(PipelineComponent):
    def __init__(self, name: str, value: int) -> None:
        super().__init__(name=name)
        self.value = value

    def run(self, context: PipelineContext) -> ComponentResult:
        context.state[self.name] = self.value
        return ComponentResult(
            component=self.name,
            passed=True,
            summary={"value": self.value},
            artifacts={"marker": context.artifact_path(f"{self.name}.json")},
        )


def test_pipeline_context_creates_artifact_paths_without_touching_files(tmp_path: Path) -> None:
    context = PipelineContext(artifact_dir=tmp_path, state={"seed": 7})

    path = context.artifact_path("stage/result.json")

    assert path == tmp_path / "stage" / "result.json"
    assert path.parent.exists()
    assert not path.exists()
    assert context.state["seed"] == 7


def test_full_pipeline_runs_components_in_order_and_collects_manifests(tmp_path: Path) -> None:
    pipeline = FullPipeline.from_components(
        [
            RecordingComponent("first", 1),
            RecordingComponent("second", 2),
        ],
        artifact_dir=tmp_path,
    )

    result = pipeline.run()

    assert result.passed is True
    assert result.context.state == {"first": 1, "second": 2}
    assert [item.component for item in result.components] == ["first", "second"]
    assert result.manifest["pipeline"] == "FullPipeline"
    assert result.manifest["passed"] is True
    assert result.manifest["components"][0]["summary"] == {"value": 1}


def test_full_pipeline_stops_when_fail_fast_component_fails(tmp_path: Path) -> None:
    class FailingComponent(PipelineComponent):
        def __init__(self) -> None:
            super().__init__(name="failing")

        def run(self, context: PipelineContext) -> ComponentResult:
            return ComponentResult(component=self.name, passed=False, summary={"reason": "boom"})

    pipeline = FullPipeline.from_components(
        [
            FailingComponent(),
            RecordingComponent("never", 99),
        ],
        artifact_dir=tmp_path,
        fail_fast=True,
    )

    result = pipeline.run()

    assert result.passed is False
    assert [item.component for item in result.components] == ["failing"]
    assert "never" not in result.context.state
