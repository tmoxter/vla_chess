import pytest


def _registered_marker_names(config: pytest.Config) -> set[str]:
    markers: list[str] = config.getini("markers")
    return {line.split(":", 1)[0].split("(", 1)[0].strip() for line in markers}


@pytest.mark.parametrize("marker", ["slow", "render"])
def test_opt_in_marker_is_registered(pytestconfig: pytest.Config, marker: str) -> None:
    """The CPU-expensive test markers are declared in the pytest configuration."""
    assert marker in _registered_marker_names(pytestconfig)


def test_unknown_markers_are_rejected(pytestconfig: pytest.Config) -> None:
    """Strict marker checking is enabled, so typo'd markers fail rather than pass silently."""
    assert pytestconfig.getoption("strict_markers") is True
