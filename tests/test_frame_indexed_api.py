"""Frame-indexed namespace API for the core catalog (§spec:render-model).

A pattern is a pure function: parameters and a frame index in, an array
out, rendered through a caller-supplied array namespace with optional
device placement. Stills ignore the frame index. The torch leg runs
only where torch is installed — the numpy leg never reaches the
backend's ``device`` and ``to`` branches, so that leg is the torch
test's alone.
"""

import numpy as np
import pytest

from display_patterns import ROI, ColorRangeError, checkerboard
from tests.conftest import assert_backend_matches_numpy, device_or_skip, to_host

SOLID_12BIT = [[4095, 2048, 0]]


class TestCheckerboard:
    """The pure checkerboard entry point."""

    def test_delivers_exact_code_values(self) -> None:
        """A 12-bit fill authored at (4095, 2048, 0) contains exactly
        those integers (§req:success-criteria exactness)."""
        frame = checkerboard(SOLID_12BIT, width=64, height=64, bit_depth=12)

        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint16
        assert set(np.unique(frame)) == {0, 2048, 4095}

    def test_still_ignores_the_frame_index(self) -> None:
        """Rendering the same still at two frame indices yields identical
        arrays — stills are pure functions of their parameters alone."""
        colors = [[4095, 4095, 4095], [0, 0, 0]]
        first = checkerboard(colors, width=64, height=48, bit_depth=12, frame=0)
        later = checkerboard(colors, width=64, height=48, bit_depth=12, frame=123)

        np.testing.assert_array_equal(first, later)

    def test_two_color_expansion_alternates(self) -> None:
        """Two colors expand to the established alternating layout."""
        frame = checkerboard([[255, 0, 0], [0, 255, 0]], width=4, height=4, bit_depth=8)

        np.testing.assert_array_equal(frame[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(frame[0, 1], [0, 255, 0])
        np.testing.assert_array_equal(frame[1, 0], [0, 255, 0])
        np.testing.assert_array_equal(frame[1, 1], [255, 0, 0])

    def test_roi_bounds_the_pattern(self) -> None:
        """Pixels outside the region of interest stay black."""
        roi = ROI(x=2, y=2, width=4, height=4)
        frame = checkerboard([[255, 255, 255]], width=8, height=8, bit_depth=8, roi=roi)

        np.testing.assert_array_equal(frame[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(frame[2, 2], [255, 255, 255])
        np.testing.assert_array_equal(frame[5, 5], [255, 255, 255])
        np.testing.assert_array_equal(frame[6, 6], [0, 0, 0])

    def test_tile_parity_registers_to_the_roi_origin(self) -> None:
        """The tile's parity runs from the region's origin, not the
        frame's, so an odd-origin region still starts on colour zero.
        Pinned because the render builds rows by parity rather than
        writing strided slices, and the two must agree."""
        roi = ROI(x=1, y=1, width=3, height=3)
        frame = checkerboard(
            [[10, 10, 10], [20, 20, 20]], width=5, height=5, bit_depth=8, roi=roi
        )

        np.testing.assert_array_equal(
            frame[..., 0],
            [
                [0, 0, 0, 0, 0],
                [0, 10, 20, 10, 0],
                [0, 20, 10, 20, 0],
                [0, 10, 20, 10, 0],
                [0, 0, 0, 0, 0],
            ],
        )

    def test_a_roi_reaching_past_the_frame_is_clipped(self) -> None:
        """A region wider than the frame renders to the frame's edge
        rather than raising or wrapping."""
        roi = ROI(x=3, y=3, width=99, height=99)
        frame = checkerboard([[7, 7, 7]], width=5, height=5, bit_depth=8, roi=roi)

        assert frame.shape == (5, 5, 3)
        np.testing.assert_array_equal(frame[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(frame[4, 4], [7, 7, 7])

    def test_out_of_range_color_raises(self) -> None:
        """Values above the stated bit depth raise the specific error."""
        with pytest.raises(ColorRangeError):
            checkerboard([[4096, 0, 0]], width=8, height=8, bit_depth=12)

    def test_negative_color_raises(self) -> None:
        with pytest.raises(ColorRangeError):
            checkerboard([[-1, 0, 0]], width=8, height=8, bit_depth=12)

    def test_bad_color_shape_raises(self) -> None:
        with pytest.raises(RuntimeError, match="shape"):
            checkerboard([[0, 0], [1, 1]], width=8, height=8, bit_depth=8)

    def test_matches_legacy_pattern_generator(self) -> None:
        """The legacy class surface and the pure entry point render the
        same array for the same parameters."""
        from display_patterns.image_generators.checkerboard import (
            DEFAULT_PATTERN_BUFFER,
            DEFAULT_PATTERN_GENERATOR,
        )

        generator = DEFAULT_PATTERN_GENERATOR
        colors = [[2000, 2000, 2000], [0, 0, 0]]
        pure = checkerboard(
            colors,
            width=generator.width,
            height=generator.height,
            bit_depth=generator.bit_depth,
            roi=generator.roi,
        )

        assert pure.dtype == DEFAULT_PATTERN_BUFFER.dtype
        np.testing.assert_array_equal(pure, DEFAULT_PATTERN_BUFFER)


def test_checkerboard_renders_identically_under_torch() -> None:
    """The same parameters render the same values through torch's
    namespace (§req:success-criteria backends). Skips where torch is
    absent — this leg alone reaches the backend's device branches."""
    assert_backend_matches_numpy(
        checkerboard, SOLID_12BIT, width=32, height=32, bit_depth=12
    )


class TestCheckerboardDtype:
    """Output dtype is the caller's, defaulting to the exact-integer
    type the pattern has always returned (§spec:backend-portability)."""

    def test_defaults_to_uint16(self) -> None:
        frame = checkerboard(SOLID_12BIT, width=8, height=8, bit_depth=12)
        assert frame.dtype == np.uint16

    def test_renders_at_a_caller_stated_dtype(self) -> None:
        """A wider integer carries the code values just as exactly."""
        frame = checkerboard(
            SOLID_12BIT, width=8, height=8, bit_depth=12, dtype=np.int32
        )
        assert frame.dtype == np.int32
        assert set(np.unique(frame)) == {0, 2048, 4095}

    def test_float_dtype_carrying_the_depth_exactly_is_accepted(self) -> None:
        """float32 represents every 12-bit code value exactly."""
        frame = checkerboard(
            SOLID_12BIT, width=8, height=8, bit_depth=12, dtype=np.float32
        )
        assert frame.dtype == np.float32
        assert set(np.unique(frame)) == {0.0, 2048.0, 4095.0}

    def test_rejects_a_dtype_too_narrow_for_the_bit_depth(self) -> None:
        """uint8 cannot hold a 12-bit code value, so it is refused
        rather than silently wrapping."""
        with pytest.raises(ValueError, match="cannot represent"):
            checkerboard(SOLID_12BIT, width=8, height=8, bit_depth=12, dtype=np.uint8)

    def test_rejects_a_float16_too_narrow_for_the_bit_depth(self) -> None:
        """float16 represents integers exactly only to 2048, short of a
        12-bit maximum."""
        with pytest.raises(ValueError, match="cannot represent"):
            checkerboard(SOLID_12BIT, width=8, height=8, bit_depth=12, dtype=np.float16)


@pytest.mark.parametrize(
    "device_name",
    [
        pytest.param("cuda", marks=pytest.mark.cuda),
        pytest.param("mps", marks=pytest.mark.mps),
    ],
)
def test_checkerboard_renders_on_a_device(device_name: str) -> None:
    """A checkerboard renders on a device at a caller-stated dtype
    (§spec:backend-portability). MPS has no working uint16, so the
    caller states int32 and the values arrive unchanged."""
    torch = pytest.importorskip("torch")
    device = device_or_skip(device_name)
    frame = checkerboard(
        SOLID_12BIT,
        width=32,
        height=32,
        bit_depth=12,
        xp=torch,
        device=device,
        dtype=torch.int32,
    )
    assert frame.device.type == device_name
    assert set(np.unique(to_host(frame))) == {0, 2048, 4095}


@pytest.mark.mps
def test_a_caller_stated_dtype_is_what_compiles_on_mps() -> None:
    """The dtype parameter is what keeps the fused path open on MPS.

    uint16 renders there in eager mode, but Inductor's Metal backend
    has no uint16 in its dtype table, so a uint16 output cannot be
    compiled. An int32 output can (§spec:backend-portability). This is
    a property of the backend, not of this library — it is asserted
    here so the reason the parameter exists stays checkable.
    """
    torch = pytest.importorskip("torch")
    device = device_or_skip("mps")
    render = torch.compile(checkerboard, dynamic=False)
    kwargs = {"width": 32, "height": 32, "bit_depth": 12, "xp": torch, "device": device}

    frame = render(SOLID_12BIT, dtype=torch.int32, **kwargs)
    assert set(np.unique(to_host(frame))) == {0, 2048, 4095}

    with pytest.raises(Exception, match="uint16"):
        torch.compile(checkerboard, dynamic=False)(
            SOLID_12BIT, dtype=torch.uint16, **kwargs
        )
