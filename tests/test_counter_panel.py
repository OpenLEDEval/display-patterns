"""Temporal-alignment counter panel codec (§spec:catalog).

Encode and decode ship together: a panel rendered at frame N decodes
back to N from the array, and after a lossy trip, because decode
samples each cell's centre and thresholds at the value midpoint. The
math is ported from backlit_molecule's probe (its ``§spec:alignment-probe``)
onto the frame-indexed namespace signature; these tests carry the
numpy leg, with a torch leg (the only one reaching the backend's
device branches) that skips where torch is absent.
"""

import numpy as np
import pytest

from display_patterns import PanelGeometry, decode_counter, render_counter_panel
from tests.conftest import assert_backend_matches_numpy


def _geometry(bits: int) -> PanelGeometry:
    return PanelGeometry.for_frame(width=64, height=16, bits=bits)


def test_render_then_decode_round_trips_the_frame_index() -> None:
    """A panel rendered at frame N decodes back to N — the counter
    survives the render as recoverable binary bit-cells."""
    geom = _geometry(bits=8)

    for frame in (0, 1, 2, 5, 42, 255):
        overlay, _mask = render_counter_panel(frame, geom)
        assert decode_counter(overlay, geom) == frame


def test_counter_wraps_modulo_two_to_the_bits() -> None:
    """The counter rolls over at 2**bits — a frame index above the range
    decodes to its low ``bits``, exactly as a fixed-width counter wraps."""
    geom = _geometry(bits=4)  # range 0..15

    overlay, _mask = render_counter_panel(16 + 3, geom)
    assert decode_counter(overlay, geom) == 3


def test_rendering_is_a_pure_function_of_the_frame_index() -> None:
    """Two calls with the same frame index produce identical arrays —
    no hidden state (§spec:render-model)."""
    geom = _geometry(bits=8)

    first_overlay, first_mask = render_counter_panel(42, geom)
    again_overlay, again_mask = render_counter_panel(42, geom)
    np.testing.assert_array_equal(first_overlay, again_overlay)
    np.testing.assert_array_equal(first_mask, again_mask)


def test_overlay_is_hwc_and_mask_is_single_channel() -> None:
    """Overlay is an (H, W, 3) float32 frame in [0, 1]; mask is an
    (H, W) float32 matte — the library's array layout, not the source
    runtime's NCHW."""
    geom = _geometry(bits=8)

    overlay, mask = render_counter_panel(7, geom)
    assert overlay.shape == (16, 64, 3)
    assert mask.shape == (16, 64)
    assert overlay.dtype == np.float32
    assert mask.dtype == np.float32
    assert float(overlay.min()) >= 0.0
    assert float(overlay.max()) <= 1.0


def test_mask_marks_the_panel_and_leaves_the_border_unmasked() -> None:
    """The mask is 1 over the panel's bounding box and 0 outside it, so
    a downstream composite touches only the panel region."""
    geom = _geometry(bits=8)

    _overlay, mask = render_counter_panel(255, geom)
    # Top-left corner is outside the title-safe panel: unmasked.
    assert float(mask[0, 0]) == 0.0
    # A cell centre is inside the panel: masked.
    row, col = geom.cell_centre(0)
    assert float(mask[row, col]) == 1.0


def test_decode_survives_a_lossy_chain() -> None:
    """Midpoint thresholding recovers the frame index from an overlay
    degraded by gain and lift, as a real video chain degrades it."""
    geom = _geometry(bits=8)

    overlay, _mask = render_counter_panel(0b10110101, geom)
    degraded = overlay * 0.7 + 0.1  # lit cells 0.8, unlit 0.1
    assert decode_counter(degraded, geom) == 0b10110101


def test_geometry_rejects_more_bits_than_columns() -> None:
    """A counter too wide for the frame has sub-pixel cells — reject it
    rather than render an undecodable panel."""
    with pytest.raises(ValueError, match="cell"):
        PanelGeometry.for_frame(width=8, height=8, bits=64)


def test_geometry_rejects_oversized_frame() -> None:
    """An out-of-range dimension would request an enormous allocation —
    reject it with a clear range error before rendering."""
    with pytest.raises(ValueError, match="out of range"):
        PanelGeometry.for_frame(width=200_000, height=2160, bits=16)
    with pytest.raises(ValueError, match="out of range"):
        PanelGeometry.for_frame(width=3840, height=0, bits=16)


def test_round_trip_under_torch() -> None:
    """The same frame index renders and decodes identically through
    torch's namespace. Skips where torch is absent — this leg alone
    reaches the backend's device branches."""
    geom = _geometry(bits=8)
    assert_backend_matches_numpy(render_counter_panel, 42, geom)


def test_geometry_rejects_a_zero_bit_counter() -> None:
    """A counter needs at least one bit-cell; ``bits=0`` is rejected
    with a range error rather than a division crash."""
    with pytest.raises(ValueError, match="at least one"):
        PanelGeometry.for_frame(width=64, height=16, bits=0)
