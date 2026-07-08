from __future__ import annotations

import tomllib
from pathlib import Path


def test_plotting_is_a_core_package_dependency() -> None:
    project_root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]

    assert project["version"] == "0.3.3"
    assert "matplotlib>=3.9" in project["dependencies"]
    assert "scikit-learn>=1.5" in project["dependencies"]
    assert "plots" not in project.get("optional-dependencies", {})
