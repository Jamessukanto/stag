"""Exercises the import-linter contracts so module boundaries fail in pytest.

This runs the same contracts declared in ``pyproject.toml`` (notably the hard
"eval must never import models" invariant) as part of the normal test run, not
only in a separate CI step.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest


@pytest.mark.skipif(
    shutil.which("lint-imports") is None,
    reason="import-linter not installed (pip install -e '.[dev]')",
)
def test_import_contracts_hold() -> None:
    result = subprocess.run(
        ["lint-imports"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "import-linter contracts violated:\n"
        f"{result.stdout}\n{result.stderr}"
    )
