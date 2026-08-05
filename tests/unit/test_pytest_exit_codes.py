import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_pytest(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_fully_deselected_run_exits_zero() -> None:
    """A marker filter that discards every collected test reports success."""
    result = _run_pytest("-m", "slow", "tests")
    assert result.returncode == pytest.ExitCode.OK, result.stdout


def test_run_without_any_collected_test_fails() -> None:
    """A suite that collects nothing at all still fails, so the gate cannot pass empty."""
    result = _run_pytest("-m", "not slow", "tests/unit", "--ignore-glob=tests/unit/test_*.py")
    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, result.stdout
