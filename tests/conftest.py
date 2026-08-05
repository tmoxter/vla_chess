from collections.abc import Sequence

import pytest

_DESELECTED = pytest.StashKey[int]()


def pytest_deselected(items: Sequence[pytest.Item]) -> None:
    """Record how many collected tests a filter expression discarded."""
    if items:
        stash = items[0].config.stash
        stash[_DESELECTED] = stash.get(_DESELECTED, 0) + len(items)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Treat a run whose collected tests were all deselected as success, but keep an empty
    suite a failure."""
    if exitstatus != pytest.ExitCode.NO_TESTS_COLLECTED:
        return
    if session.config.stash.get(_DESELECTED, 0) > 0:
        session.exitstatus = pytest.ExitCode.OK
