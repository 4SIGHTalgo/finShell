from __future__ import annotations

import subprocess
from pathlib import Path


FORBIDDEN_INTERNAL_TERMS = (
    "agentic",
    "chain of thought",
    "codex",
    "council",
    "plugin",
    "superpowers",
)


def test_tracked_markdown_is_exclusively_library_documentation() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    violations: list[str] = []
    for relative_path in tracked:
        path = repo_root / relative_path
        text = path.read_text(encoding="utf-8").lower()
        matched = [term for term in FORBIDDEN_INTERNAL_TERMS if term in text]
        if matched:
            violations.append(f"{relative_path}: {', '.join(matched)}")

    assert not violations, "internal agent documentation is tracked:\n" + "\n".join(violations)
