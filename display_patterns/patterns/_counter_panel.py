"""Temporal-alignment counter panel: geometry, encode, decode.

The frame index is rendered as binary bit-cells in one title-safe row,
MSB-first, with the matching decoder (§spec:catalog). Geometry is
deterministic — encoder and decoder agree on every cell from
parameters alone — and decode samples each cell's centre, thresholding
at the value midpoint, so the counter survives a lossy video chain.
Ported from backlit_molecule's probe math (its ``§spec:alignment-probe``)
onto the frame-indexed namespace signature.

Value and layout conventions (§spec:render-model): this is a float
pattern; the overlay is ``(height, width, 3)`` float32 in [0, 1] — a
lit cell 1.0, an unlit cell 0.0 — and the mask ``(height, width)``
float32, 1 over the panel's bounding box. HWC replaces the source
runtime's batched NCHW because this library's arrays are plain images,
not graph payloads; the mask ships so a consumer can composite the
panel over existing content and leave the rest untouched. Meaning
stays with the caller — a consumer driving integer code values scales
[0, 1] into its own value space and normalizes captures back before
decoding.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from display_patterns.patterns import _backend

# Upper bound on either frame dimension. Geometry sizes two float32
# arrays from caller parameters, so an out-of-range value would request
# an enormous allocation. 16384 clears 8K (7680x4320) with headroom;
# beyond it is not a real video frame, so reject it with a clear error
# rather than fail on the allocation.
_MAX_DIMENSION = 16384

# Fraction of each frame dimension the panel insets from the edge, so
# the bit-cells land inside title-safe and survive overscan.
_TITLE_SAFE_FRACTION = 0.1

# Bit-cell values: a lit cell is 1.0, an unlit cell 0.0 (float range
# convention [0, 1]). Decode thresholds at the midpoint so gain and
# lift short of half-scale cannot flip a bit.
_CELL_ON = 1.0
_DECODE_THRESHOLD = 0.5


@dataclass(frozen=True)
class PanelGeometry:
    """Where the bit-cells land in the frame, derived from frame size
    and bit width. Deterministic, so :func:`render_counter_panel` and
    :func:`decode_counter` agree on every cell's pixels."""

    width: int
    height: int
    bits: int
    pad_x: int
    pad_y: int
    cell_w: int
    cell_h: int

    @classmethod
    def for_frame(cls, width: int, height: int, bits: int) -> "PanelGeometry":
        """Lay ``bits`` square-ish cells in one title-safe row across the frame.

        Raises
        ------
        ValueError
            If a dimension falls outside [1, 16384], or the frame is too
            small to give each cell at least one pixel (a sub-pixel cell
            renders an undecodable panel).
        """
        for label, value in (("width", width), ("height", height)):
            if not 1 <= value <= _MAX_DIMENSION:
                raise ValueError(
                    f"{label} {value} is out of range: a frame dimension must be "
                    f"between 1 and {_MAX_DIMENSION} pixels."
                )
        pad_x = round(width * _TITLE_SAFE_FRACTION)
        pad_y = round(height * _TITLE_SAFE_FRACTION)
        cell_w = (width - 2 * pad_x) // bits
        if cell_w < 1:
            raise ValueError(
                f"{bits} bit-cells do not fit across a {width}px frame: each "
                f"cell would be under one pixel wide. Use fewer bits or a wider "
                f"frame."
            )
        cell_h = min(cell_w, height - 2 * pad_y)
        if cell_h < 1:
            raise ValueError(
                f"a bit-cell does not fit within a {height}px frame's title-safe "
                f"region: the cell would be under one pixel tall."
            )
        return cls(width, height, bits, pad_x, pad_y, cell_w, cell_h)

    @property
    def overlay_shape(self) -> tuple[int, int, int]:
        """The ``(height, width, 3)`` shape of a rendered overlay."""
        return (self.height, self.width, 3)

    @property
    def mask_shape(self) -> tuple[int, int]:
        """The ``(height, width)`` shape of a rendered mask."""
        return (self.height, self.width)

    @property
    def panel_cols(self) -> tuple[int, int]:
        """The panel's horizontal bounds ``(start, stop)`` in pixels."""
        return self.pad_x, self.pad_x + self.bits * self.cell_w

    @property
    def panel_rows(self) -> tuple[int, int]:
        """The panel's vertical bounds ``(start, stop)`` in pixels."""
        return self.pad_y, self.pad_y + self.cell_h

    def cell_centre(self, index: int) -> tuple[int, int]:
        """The ``(row, col)`` pixel at the centre of bit-cell ``index``
        (MSB-first)."""
        row = self.pad_y + self.cell_h // 2
        col = self.pad_x + index * self.cell_w + self.cell_w // 2
        return row, col


def render_counter_panel(
    frame: int, geometry: PanelGeometry, *, xp: Any = np, device: Any = None
) -> tuple[Any, Any]:
    """Render frame index ``frame`` into a counter panel: ``(overlay, mask)``.

    A pure function of ``frame`` and ``geometry`` (§spec:render-model):
    the frame index is the pattern's only time source, wrapping modulo
    ``2**geometry.bits``.

    Parameters
    ----------
    frame : int
        Frame index to encode, MSB-first.
    geometry : PanelGeometry
        Cell layout; build with :meth:`PanelGeometry.for_frame`.
    xp : namespace, optional
        Array namespace to render through (numpy default; torch on GPU
        hosts).
    device : optional
        Device placement for ``xp`` backends that take one.

    Returns
    -------
    (overlay, mask)
        ``overlay`` is ``(height, width, 3)`` float32 in [0, 1] — the
        counter's bits as a row of lit (1.0) / unlit (0.0) cells,
        MSB-first, the rest 0. ``mask`` is ``(height, width)`` float32,
        1 over the panel's bounding box and 0 outside it, so a
        composite touches only the panel.
    """
    overlay = _backend.full(xp, geometry.overlay_shape, 0.0, xp.float32, device)
    mask = _backend.full(xp, geometry.mask_shape, 0.0, xp.float32, device)

    r0, r1 = geometry.panel_rows
    c0, c1 = geometry.panel_cols
    mask[r0:r1, c0:c1] = _CELL_ON

    bits = geometry.bits
    for index in range(bits):
        # MSB-first: bit 0 is the most-significant, so a wider counter
        # reads left-to-right like a written binary number.
        if (frame >> (bits - 1 - index)) & 1:
            cell_c0 = geometry.pad_x + index * geometry.cell_w
            overlay[r0:r1, cell_c0 : cell_c0 + geometry.cell_w, :] = _CELL_ON
    return overlay, mask


def decode_counter(overlay: Any, geometry: PanelGeometry) -> int:
    """Recover the frame index from a rendered ``overlay`` — the inverse
    of :func:`render_counter_panel`. Samples each cell's centre and
    thresholds at the value midpoint, so it survives a lossy round trip
    (wire encode, resample). ``overlay`` is any ``(height, width, 3)``
    array in the [0, 1] range convention, on any backend."""
    value = 0
    for index in range(geometry.bits):
        row, col = geometry.cell_centre(index)
        bit = 1 if float(overlay[row, col, 0]) >= _DECODE_THRESHOLD else 0
        value = (value << 1) | bit
    return value
