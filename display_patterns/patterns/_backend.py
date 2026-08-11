"""Duck-typed helpers over a caller-supplied array namespace.

The ``xp`` parameter is the whole backend mechanism (§spec:render-model):
numpy on CPU hosts, torch on GPU hosts. Nothing here imports a backend —
creation passes the optional ``device`` through only when the caller
supplies one (numpy accepts none by default; torch places the array),
and dtype conversion duck-types between numpy's ``astype`` and torch's
``to``.
"""

from typing import Any


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
