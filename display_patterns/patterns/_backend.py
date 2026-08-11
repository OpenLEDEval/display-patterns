"""Duck-typed helpers over a caller-supplied array namespace.

The ``xp`` parameter is the backend mechanism (§spec:render-model):
numpy on CPU hosts, torch on GPU hosts. Nothing here imports a backend.

The contract a namespace has to satisfy — recorded here because it is
the answer to "would backend X work?":

- ``zeros``/``full``/``asarray`` accepting ``dtype`` (and ``device``
  when the caller passes one; creation calls omit the keyword
  otherwise, so numpy's default signature works),
- dtype attributes ``uint16`` and ``float32``,
- **mutable arrays**: render bodies write strided slices in place, so
  an immutable-array backend (JAX) is out of contract,
- ``astype`` or ``to`` on the array for dtype conversion, and ``cpu``
  where device-resident arrays cannot be read by numpy directly.
"""

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
