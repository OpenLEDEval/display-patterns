"""Temporal-alignment counter panel: geometry, encode, decode.

The frame index is rendered as binary bit-cells in one title-safe row,
MSB-first, with the matching decoder (§spec:catalog). Geometry is
deterministic — encoder and decoder agree on every cell from
parameters alone — and decode samples each cell's centre, thresholding
at the value midpoint, so the counter survives a lossy video chain.
Ported from the alignment-probe math of a real-time LED render runtime
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

# Upper bound on the counter's width. Bit extraction runs in signed
# 32-bit integers, the widest every backend supports without qualification
# (§spec:backend-portability). At 60 Hz a 31-bit counter runs over a year
# before wrapping, so the bound costs no real counter anything.
_MAX_BITS = 31

# Row kinds in the per-row gather that lays the panel: a frame row either
# falls inside the panel's band or is blank.
_BLANK_ROW = 0
_PANEL_ROW = 1


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
            If ``bits`` falls outside [1, 31], a dimension falls outside
            [1, 16384], or the frame is too small to give each cell at
            least one pixel (a sub-pixel cell renders an undecodable
            panel).
        """
        if not 1 <= bits <= _MAX_BITS:
            raise ValueError(
                f"bits {bits} is out of range: a counter carries between 1 and "
                f"{_MAX_BITS} bit-cells."
            )
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
    frame: int | Any, geometry: PanelGeometry, *, xp: Any = np, device: Any = None
) -> tuple[Any, Any]:
    """Render frame index ``frame`` into a counter panel: ``(overlay, mask)``.

    A pure function of ``frame`` and ``geometry`` (§spec:render-model):
    the frame index is the pattern's only time source, wrapping modulo
    ``2**geometry.bits``.

    Parameters
    ----------
    frame : int or array
        Frame index to encode, MSB-first. A zero-dimensional array is
        accepted and preferred by a consumer compiling its frame loop:
        a Python integer is a compile-time constant, so stepping one
        recompiles the pattern (§spec:backend-portability). Wraps
        modulo ``2**geometry.bits``.
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
    bits = geometry.bits
    if isinstance(frame, int):
        # Wrap host-side, exactly: a Python integer is unbounded, and only
        # the low ``bits`` bits are read.
        frame = frame & ((1 << bits) - 1)
    counter = _backend.astype(_backend.asarray(xp, frame, device), xp.int32)

    r0, r1 = geometry.panel_rows
    c0, c1 = geometry.panel_cols
    cols = _backend.arange(xp, geometry.width, xp.int32, device)
    rows = _backend.arange(xp, geometry.height, xp.int32, device)

    # Which bit-cell each column falls in, and so which bit it shows.
    # MSB-first: cell 0 is the most-significant, so a wider counter reads
    # left-to-right like a written binary number.
    in_panel_cols = (cols >= c0) & (cols < c1)
    cell = (cols - geometry.pad_x) // geometry.cell_w
    # Columns outside the panel would shift by an out-of-range amount,
    # which is undefined; park them at zero and mask them out after.
    shift = xp.where(in_panel_cols, bits - 1 - cell, 0)
    lit = xp.bitwise_right_shift(counter, shift) & 1

    # A frame has two kinds of row — inside the panel's band, or blank —
    # so the raster is two rows built at row width and one gather that
    # selects between them per row, materializing each output once
    # (§spec:backend-portability).
    # Shaped (1, 3) so the where broadcasts each column across the
    # three channels, giving a (width, 3) row.
    on = _backend.full(xp, (1, 3), _CELL_ON, xp.float32, device)
    off = _backend.zeros(xp, (1, 3), xp.float32, device)
    panel_row = xp.where((in_panel_cols & (lit == 1))[:, None], on, off)
    overlay_rows = xp.stack([xp.zeros_like(panel_row), panel_row])
    mask_row = xp.where(in_panel_cols, _CELL_ON, 0.0)
    mask_row = _backend.astype(mask_row, xp.float32)
    mask_rows = xp.stack([xp.zeros_like(mask_row), mask_row])

    row_kind = xp.where((rows >= r0) & (rows < r1), _PANEL_ROW, _BLANK_ROW)
    return overlay_rows[row_kind], mask_rows[row_kind]


def decode_counter(overlay: Any, geometry: PanelGeometry) -> int:
    """Recover the frame index from a rendered ``overlay`` — the inverse
    of :func:`render_counter_panel`. Samples each cell's centre and
    thresholds at the value midpoint, so it survives a lossy round trip
    (wire encode, resample). ``overlay`` is any ``(height, width, 3)``
    array in the [0, 1] range convention, on any backend."""
    # One gather of every cell centre, one device-to-host transfer:
    # per-bit scalar reads would sync a device-resident overlay once per
    # bit (§req:success-criteria — no copy through host memory per bit).
    centres = [geometry.cell_centre(index) for index in range(geometry.bits)]
    rows = [row for row, _ in centres]
    cols = [col for _, col in centres]
    samples = _backend.to_host(overlay[rows, cols, 0])
    value = 0
    for sample in samples:
        value = (value << 1) | (1 if float(sample) >= _DECODE_THRESHOLD else 0)
    return value
