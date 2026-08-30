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

    Returns
    -------
    array
        ``(height, width, 3)`` ``uint16`` array on ``xp``.

    Raises
    ------
    RuntimeError
        If ``colors`` has an invalid shape.
    ColorRangeError
        If a color value falls outside the stated bit depth's range.
    """
    del frame  # a still ignores the frame index (§spec:render-model)

    expanded = _expand_colors(colors, bit_depth)
    if roi is None:
        roi = ROI(0, 0, width, height)

    # Write the tile colors directly into a target-dtype frame — ~3x
    # cheaper than a palette gather at 1080p — and black outside the
    # ROI comes free with the zero allocation.
    image = _backend.zeros(xp, (height, width, 3), xp.uint16, device)
    palette = _backend.astype(_backend.asarray(xp, expanded, device), xp.uint16)

    y_end = min(roi.y2, height)
    x_end = min(roi.x2, width)
    if bool(np.all(expanded == expanded[0])):
        # One distinct color: a single contiguous write renders the solid.
        image[roi.y : y_end, roi.x : x_end, :] = palette[0]
    else:
        image[roi.y : y_end : 2, roi.x : x_end : 2, :] = palette[0]
        image[roi.y + 1 : y_end : 2, roi.x : x_end : 2, :] = palette[1]
        image[roi.y : y_end : 2, roi.x + 1 : x_end : 2, :] = palette[2]
        image[roi.y + 1 : y_end : 2, roi.x + 1 : x_end : 2, :] = palette[3]
    return image
