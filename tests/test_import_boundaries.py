"""Mechanical enforcement of module import boundaries."""

from __future__ import annotations

import subprocess
import sys


def test_import_boundaries_pass() -> None:
    """Run import-linter against contracts in pyproject.toml."""
    result = subprocess.run(
        [sys.executable, "-m", "importlinter.cli"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "Import boundary violation:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
