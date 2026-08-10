"""Package-level smoke tests for the display_patterns scaffold."""

import importlib
from importlib.metadata import version


def test_import() -> None:
    """The core package imports."""
    module = importlib.import_module("display_patterns")
    assert module.__name__ == "display_patterns"


def test_distribution_metadata() -> None:
    """The distribution is installed as display-patterns."""
    assert version("display-patterns")
