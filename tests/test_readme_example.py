from __future__ import annotations

import re
from pathlib import Path

import pytest


START = "<!-- notebook-example:start -->"
END = "<!-- notebook-example:end -->"


def test_notebook_readme_cells_execute_and_match_outputs(tmp_path: Path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text(encoding="utf-8")

    assert START in readme
    assert END in readme
    section = readme.split(START, maxsplit=1)[1].split(END, maxsplit=1)[0]
    cells = re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL)
    assert len(cells) == 4
    assert section.count("```text") == 4
    combined = "\n".join(cells)
    assert combined.count("import finshell as fs") == 1
    assert "from finshell" not in combined
    assert "np.sin" not in combined
    assert "default_rng" in combined

    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__main__"}
    for index, cell in enumerate(cells, start=1):
        exec(compile(cell, f"README-cell-{index}", "exec"), namespace)

    label = namespace["label"]
    cv = namespace["cv"]
    oos = namespace["oos"]
    economics = namespace["economics"]
    assert label.passed is True
    assert label.summary["real_total"] > label.summary["random_final_p95"]
    assert cv.passed is True
    assert cv.summary["valid_bootstrap_fits"] > 0
    assert oos.passed is True
    assert oos.summary["selected_total"] > oos.summary["random_final_p95"]
    assert oos.summary["selected_total"] > oos.summary["no_selector_total"]
    assert economics.passed is True
    assert 0.0 <= economics.summary["upper_hit_probability"] <= 1.0
    assert sum(
        economics.summary[key]
        for key in (
            "upper_hit_probability",
            "lower_hit_probability",
            "vertical_probability",
        )
    ) == pytest.approx(1.0)
    assert economics.summary["median_resolution_trades"] > 0.0
    economic_diagnostics = namespace["study"].context.state["economic_study_diagnostics"]
    assert (
        economic_diagnostics["upper_balance"] - economic_diagnostics["initial_balance"]
        == economic_diagnostics["initial_balance"] - economic_diagnostics["lower_balance"]
    )

    expected_output_fragments = [
        f"real_total={label.summary['real_total']:.4f}",
        f"valid_bootstrap_fits={cv.summary['valid_bootstrap_fits']}",
        f"selected_total={oos.summary['selected_total']:.4f}",
        f"upper_hit={economics.summary['upper_hit_probability']:.1%}",
        f"lower_hit={economics.summary['lower_hit_probability']:.1%}",
        f"vertical={economics.summary['vertical_probability']:.1%}",
        f"median_resolution={economics.summary['median_resolution_trades']:.1f} trades",
    ]
    assert all(fragment in section for fragment in expected_output_fragments)

    image_urls = re.findall(r"!\[[^]]*\]\(([^)]+\.png)\)", section)
    assert len(image_urls) == 4
    assert all("raw.githubusercontent.com/4SIGHTalgo/finShell/main/" in url for url in image_urls)
    for url in image_urls:
        filename = url.rsplit("/", maxsplit=1)[1]
        local_path = project_root / "assets" / "readme" / "plots" / filename
        assert local_path.stat().st_size > 0
