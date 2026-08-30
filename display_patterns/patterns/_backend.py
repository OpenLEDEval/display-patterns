"""Duck-typed helpers over a caller-supplied array namespace.

The ``xp`` parameter is the backend mechanism (§spec:render-model):
numpy on CPU hosts, torch on GPU hosts. Nothing here imports a backend.

The contract a namespace has to satisfy — recorded here because it is
the answer to "would backend X work?":

- ``zeros``/``full``/``asarray``/``arange`` accepting ``dtype`` (and
  ``device`` when the caller passes one; creation calls omit the
  keyword otherwise, so numpy's default signature works),
- ``where``, ``stack``, ``zeros_like``, ``reshape`` on the array, and
  the arithmetic and comparison operators used to build a raster from
  coordinate arrays,
- indexing an array with an integer index array (the per-row gather),
- ``iinfo``/``finfo`` for the dtype range check,
- dtype attributes ``uint16``, ``int32``, and ``float32``,
- ``astype`` or ``to`` on the array for dtype conversion, and ``cpu``
  where device-resident arrays cannot be read by numpy directly.

Render bodies are functional (§spec:backend-portability): they build
results from broadcast arithmetic rather than writing strided slices
into an allocation, so a backend whose arrays are immutable is in
contract.
"""

import math
from typing import Any

import numpy as np


def zeros(xp: Any, shape: tuple[int, ...], dtype: Any, device: Any) -> Any:
    """A zero ``shape`` array on ``xp``, placed on ``device`` when the
    caller supplies one."""
    if device is not None:
        return xp.zeros(shape, dtype=dtype, device=device)
    return xp.zeros(shape, dtype=dtype)


def full(xp: Any, shape: tuple[int, ...], fill: Any, dtype: Any, device: Any) -> Any:
    """A ``shape`` array of ``fill`` on ``xp``, placed on ``device`` when
    the caller supplies one."""
    if device is not None:
        return xp.full(shape, fill, dtype=dtype, device=device)
    return xp.full(shape, fill, dtype=dtype)


def asarray(xp: Any, values: Any, device: Any) -> Any:
    """``values`` as an ``xp`` array, placed on ``device`` when the
    caller supplies one."""
    if device is not None:
        return xp.asarray(values, device=device)
    return xp.asarray(values)


def arange(xp: Any, stop: int, dtype: Any, device: Any) -> Any:
    """``[0, stop)`` as an ``xp`` array, placed on ``device`` when the
    caller supplies one."""
    if device is not None:
        return xp.arange(stop, dtype=dtype, device=device)
    return xp.arange(stop, dtype=dtype)


def max_exact_integer(xp: Any, dtype: Any) -> int:
    """The largest integer ``dtype`` represents without loss.

    Integer dtypes report their own maximum. A float dtype represents
    consecutive integers only up to two to the power of its mantissa
    width plus one, which its epsilon states — both namespaces expose
    ``finfo.eps`` where neither exposes a mantissa width.
    """
    try:
        return int(xp.iinfo(dtype).max)
    except (TypeError, ValueError):
        eps = float(xp.finfo(dtype).eps)
        return 2 ** (round(-math.log2(eps)) + 1)


def astype(array: Any, dtype: Any) -> Any:
    """``array`` converted to ``dtype`` — numpy's ``astype`` or torch's
    ``to``, whichever the array carries."""
    if hasattr(array, "astype"):
        return array.astype(dtype)
    return array.to(dtype)


def to_host(array: Any) -> np.ndarray:
    """``array`` as a host numpy array in one device-to-host transfer."""
    if hasattr(array, "cpu"):
        array = array.cpu()
    return np.asarray(array)
