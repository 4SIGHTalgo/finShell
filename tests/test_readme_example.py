from __future__ import annotations

import re
from pathlib import Path


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
    assert economics.summary["terminal_p50"] > 0.0

    expected_output_fragments = [
        f"real_total={label.summary['real_total']:.4f}",
        f"valid_bootstrap_fits={cv.summary['valid_bootstrap_fits']}",
        f"selected_total={oos.summary['selected_total']:.4f}",
        f"terminal_p50={economics.summary['terminal_p50']:.4f}",
    ]
    assert all(fragment in section for fragment in expected_output_fragments)

    image_paths = re.findall(r"!\[[^]]*\]\((assets/readme/[^)]+\.png)\)", section)
    assert image_paths == [
        "assets/readme/plots/01_label_audit.png",
        "assets/readme/plots/02_cpcv_selector.png",
        "assets/readme/plots/03_oos_audit.png",
        "assets/readme/plots/04_economic_monte_carlo.png",
    ]
    assert all((project_root / relative_path).stat().st_size > 0 for relative_path in image_paths)
