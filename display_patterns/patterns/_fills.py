"""Solid and checkerboard fills on the frame-indexed signature.

One entry point, :func:`checkerboard`, with the established color
expansion (§spec:catalog): one color renders a solid fill, two
alternate, four map to the 2x2 tile. Integer code values are delivered
unmodified and validated against the stated bit depth
(§spec:render-model exactness); an optional region of interest bounds
the pattern, with black outside it.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from display_patterns.patterns import _backend


@dataclass
class ROI:
    """Region of Interest for pattern generation.

    Defines a rectangular region within an image where patterns will be
    drawn. Width and height define the size of the rectangular region.

    Parameters
    ----------
    x : int, optional
        X coordinate of the top-left corner. Default is 0.
    y : int, optional
        Y coordinate of the top-left corner. Default is 0.
    width : int, optional
        Width of the ROI in pixels. Default is 100.
    height : int, optional
        Height of the ROI in pixels. Default is 100.

    Examples
    --------
    Create a centered ROI:

    >>> roi = ROI(x=100, y=100, width=800, height=600)
    >>> print(f"ROI: {roi.width}x{roi.height} at ({roi.x}, {roi.y})")
    ROI: 800x600 at (100, 100)
    """

    x: int = 0
    y: int = 0
    width: int = 100
    height: int = 100

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


# Index of the blank row in the per-row kind table: a row outside the
# region of interest selects it, so the region is bounded by the same
# gather that lays the tile rather than by a second masking pass.
_BLANK_ROW = 2


class ColorRangeError(RuntimeError):
    """Exception raised when color values are outside the valid range.

    Raised when color values exceed the maximum value for the stated
    bit depth (e.g., values > 255 for 8-bit, > 4095 for 12-bit) or fall
    below zero.

    Parameters
    ----------
    detail_string : str, optional
        Additional detail about the validation error.
    """

    def __init__(self, detail_string: str | None = None):
        super().__init__("colors must be a required bit-depth range.")
        if detail_string:
            self.add_note(detail_string)


def _expand_colors(colors: ArrayLike, bit_depth: int) -> np.ndarray:
    """Validate ``colors`` and expand 1-4 colors to the 2x2 tile.

    Expansion rules (§spec:catalog): one color fills all four squares
    (a solid), two alternate, three map to (c0, c1, c2, c0), four map
    directly.
    """
    host = np.asarray(colors)
    if host.shape == (3,):
        host = host.reshape((1, 3))
    if host.ndim != 2 or not 1 <= host.shape[0] <= 4 or host.shape[1] != 3:
        raise RuntimeError(
            "Colors must have shape (1,3) to (4,3), or single color shape (3,)"
        )

    num_colors = host.shape[0]
    if num_colors == 1:
        # Copied, not a broadcast view: a view is read-only, and torch
        # warns that it cannot back a tensor with non-writable memory.
        host = np.broadcast_to(host, (4, 3)).copy()
    elif num_colors == 2:
        host = host[(0, 1, 1, 0), :]
    elif num_colors == 3:
        host = host[(0, 1, 2, 0), :]

    # Validation runs host-side: a parameter check, not the render path,
    # so it uses numpy regardless of ``xp``.
    if not bool(np.all((host >= 0) & (host <= 2**bit_depth - 1))):
        raise ColorRangeError(f"Bit depth: {bit_depth:d}")
    return host


def _validate_dtype(xp: Any, dtype: Any, bit_depth: int) -> None:
    """Refuse a dtype that cannot carry the stated bit depth exactly.

    Exactness is preserved by this check rather than by a fixed return
    type (§spec:backend-portability), so the caller may state whatever
    its value space and its backend's compiler need.
    """
    required = 2**bit_depth - 1
    carried = _backend.max_exact_integer(xp, dtype)
    if carried < required:
        raise ValueError(
            f"dtype {dtype} cannot represent {bit_depth}-bit code values: it "
            f"carries integers exactly to {carried}, short of {required}. "
            f"State a wider dtype."
        )


def checkerboard(
    colors: ArrayLike,
    *,
    width: int,
    height: int,
    bit_depth: int,
    roi: ROI | None = None,
    frame: int = 0,
    xp: Any = np,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Render a checkerboard (or solid) fill at exact code values.

    A pure function of its parameters (§spec:render-model): the same
    inputs produce identical arrays on every run and backend. The
    requested code values arrive in the array unmodified.

    Parameters
    ----------
    colors : ArrayLike
        1-4 RGB colors; shape (3,), or (N, 3) with N in 1..4. One color
        renders a solid fill; two alternate; three map to
        (c0, c1, c2, c0); four map directly to the 2x2 tile.
    width, height : int
        Frame size in pixels.
    bit_depth : int
        Stated bit depth; every color value is validated against
        [0, 2**bit_depth - 1].
    roi : ROI, optional
        Region of interest bounding the pattern; pixels outside it are
        black. Default is the full frame.
    frame : int, optional
        Frame index. A still, this pattern ignores it — the parameter
        exists so every catalog entry shares the rendering signature.
    xp : namespace, optional
        Array namespace to render through (numpy default; torch on GPU
        hosts).
    device : optional
        Device placement for ``xp`` backends that take one; ``None``
        uses the backend default.
    dtype : optional
        Output dtype, defaulting to the namespace's ``uint16``. Any
        dtype carrying ``bit_depth`` exactly is accepted; state a wider
        one where a backend cannot compile ``uint16``
        (§spec:backend-portability).

    Returns
    -------
    array
        ``(height, width, 3)`` array on ``xp``, of ``dtype``.

    Raises
    ------
    RuntimeError
        If ``colors`` has an invalid shape.
    ColorRangeError
        If a color value falls outside the stated bit depth's range.
    ValueError
        If ``dtype`` cannot represent the stated bit depth exactly.
    """
    del frame  # a still ignores the frame index (§spec:render-model)

    expanded = _expand_colors(colors, bit_depth)
    if roi is None:
        roi = ROI(0, 0, width, height)
    if dtype is None:
        dtype = xp.uint16
    _validate_dtype(xp, dtype, bit_depth)

    # Built from broadcast arithmetic rather than written as strided slices
    # into an allocation: a scatter is a materialized intermediate no
    # compiler fuses away, and mutation puts immutable-array backends out
    # of contract (§spec:backend-portability).
    #
    # A frame has only three kinds of row — even tile row, odd tile row,
    # and blank — so the whole raster is three rows built at row width and
    # one gather that selects among them per row. That keeps every
    # elementwise pass at row scale and materializes the frame exactly
    # once; masking the region of interest afterwards would cost a second
    # full-frame pass, which measured an order of magnitude more than the
    # gather.
    palette = _backend.astype(_backend.asarray(xp, expanded, device), dtype)
    cols = _backend.arange(xp, width, xp.int32, device).reshape(width, 1)
    rows = _backend.arange(xp, height, xp.int32, device)
    blank = _backend.zeros(xp, (1, 1), dtype, device)

    # Tile parity runs from the region's origin, so the pattern registers
    # to the region rather than to the frame.
    col_odd = (cols - roi.x) % 2 == 1
    col_inside = (cols >= roi.x) & (cols < min(roi.x2, width))
    even_row = xp.where(col_inside, xp.where(col_odd, palette[2], palette[0]), blank)
    odd_row = xp.where(col_inside, xp.where(col_odd, palette[3], palette[1]), blank)
    rows_by_kind = xp.stack([even_row, odd_row, xp.zeros_like(even_row)])

    row_inside = (rows >= roi.y) & (rows < min(roi.y2, height))
    return rows_by_kind[xp.where(row_inside, (rows - roi.y) % 2, _BLANK_ROW)]
