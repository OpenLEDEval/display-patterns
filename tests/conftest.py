"""Shared test scaffolding for display-patterns."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest


def torch_or_skip() -> Any:
    """The torch module, or skip where it is absent.

    Torch is a development dependency, never a dependency of the package
    — the array namespace is duck-typed (§spec:render-model). It is
    installed for the test run so backend equivalence is a gate rather
    than a leg that skips silently (§spec:backend-portability).
    """
    return pytest.importorskip("torch")


def device_or_skip(name: str) -> Any:
    """A torch device named ``name``, or skip where the host has none.

    Guards the ``cuda`` and ``mps`` legs, which stay deselected unless
    asked for by marker: CI asserts the backend contract on torch's CPU
    build, hardware asserts the device (§spec:backend-portability).
    """
    torch = torch_or_skip()
    is_available = {
        "cuda": lambda: torch.cuda.is_available(),
        "mps": lambda: torch.backends.mps.is_available(),
    }[name]
    if not is_available():
        pytest.skip(f"host has no {name} device")
    return torch.device(name)


def to_host(array: Any) -> np.ndarray:
    """``array`` as a host numpy array, wherever it was rendered.

    Deliberately not the library's own ``_backend.to_host``: comparing
    a rendered array through the same transfer helper the render path
    used would hide a fault in that helper from every test that relies
    on it.
    """
    if hasattr(array, "cpu"):
        array = array.cpu()
    return np.asarray(array)


def decode_on_device(
    render: Callable[..., Any], *args: Any, device: Any, **kwargs: Any
) -> int:
    """Render on ``device`` through torch and decode the result there.

    The decode reads the device-resident overlay directly, so the leg
    exercises the one-transfer readback rather than a host copy the
    test made first (§spec:backend-portability).
    """
    from display_patterns import decode_counter

    torch = torch_or_skip()
    overlay, _mask = render(*args, xp=torch, device=device, **kwargs)
    geometry = args[-1]
    return decode_counter(overlay, geometry)


def assert_backend_matches_numpy(
    render: Callable[..., Any], *args: Any, device: Any = None, **kwargs: Any
) -> None:
    """Render under torch and assert equality with the numpy result.

    Handles entry points that return one array or a tuple of arrays.
    The numpy leg does not reach the backend's ``device``/``to``
    branches; only the torch leg does, and only a ``device`` argument
    reaches placement.
    """
    torch = torch_or_skip()
    expected = render(*args, **kwargs)
    actual = render(*args, xp=torch, device=device, **kwargs)
    expected_items = expected if isinstance(expected, tuple) else (expected,)
    actual_items = actual if isinstance(actual, tuple) else (actual,)
    for actual_item, expected_item in zip(actual_items, expected_items, strict=True):
        np.testing.assert_array_equal(to_host(actual_item), to_host(expected_item))
