from __future__ import annotations

from pathlib import Path


START = "<!-- full-pipeline-example:start -->"
END = "<!-- full-pipeline-example:end -->"


def test_full_readme_example_executes_pass_and_fail_candidates(tmp_path: Path, monkeypatch) -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert START in readme
    assert END in readme
    section = readme.split(START, maxsplit=1)[1].split(END, maxsplit=1)[0]
    assert "```python" in section
    code = section.split("```python", maxsplit=1)[1].split("```", maxsplit=1)[0]
    assert "class DummyModel" in code
    assert "def fit(" not in code

    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(compile(code, "README.md", "exec"), namespace)

    reports = namespace["reports"]
    assert reports["passing_candidate"]["promoted"] is True
    assert reports["failing_candidate"]["promoted"] is False
    assert reports["failing_candidate"]["failed_gates"]
    for candidate in reports:
        plot_dir = tmp_path / "outputs" / candidate / "plots"
        assert {path.name for path in plot_dir.glob("*.png")} == {
            "label_diagnostics.png",
            "cpcv_distributions.png",
            "null_test_equity_paths.png",
            "triple_barrier_bootstrap_paths.png",
        }
