from pathlib import Path

import vla_chess


def test_package_is_importable() -> None:
    """The vla_chess package is importable from the environment."""
    assert vla_chess.__name__ == "vla_chess"


def test_package_ships_py_typed_marker() -> None:
    """The package directory contains a PEP 561 py.typed marker."""
    assert vla_chess.__file__ is not None
    package_dir = Path(vla_chess.__file__).parent
    assert (package_dir / "py.typed").is_file()
