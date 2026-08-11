"""Shared test scaffolding for display-patterns."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest


def assert_backend_matches_numpy(
    render: Callable[..., Any], *args: Any, **kwargs: Any
) -> None:
    """Render under torch and assert equality with the numpy result.

    Skips where torch is absent (torch is never a dependency of this
    package — the namespace is duck-typed). Handles entry points that
    return one array or a tuple of arrays. Note the numpy leg does not
    reach the ``device``/``to`` backend branches; only this helper does,
    where torch is installed.
    """
    torch = pytest.importorskip("torch")
    expected = render(*args, **kwargs)
    actual = render(*args, xp=torch, **kwargs)
    expected_items = expected if isinstance(expected, tuple) else (expected,)
    actual_items = actual if isinstance(actual, tuple) else (actual,)
    for actual_item, expected_item in zip(actual_items, expected_items, strict=True):
        np.testing.assert_array_equal(np.asarray(actual_item), expected_item)
