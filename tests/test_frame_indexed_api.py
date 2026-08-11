"""Frame-indexed namespace API for the core catalog (§spec:render-model).

A pattern is a pure function: parameters and a frame index in, an array
out, rendered through a caller-supplied array namespace with optional
device placement. Stills ignore the frame index. The torch leg runs
only where torch is installed; the numpy leg carries the full signal
because the render path is duck-typed over the namespace.
"""

import numpy as np
import pytest

from display_patterns import ROI, ColorRangeError, checkerboard

TRICOLOR_12BIT = [[4095, 2048, 0]]


class TestCheckerboard:
    """The pure checkerboard entry point."""

    def test_delivers_exact_code_values(self) -> None:
        """A 12-bit fill authored at (4095, 2048, 0) contains exactly
        those integers (§req:success-criteria exactness)."""
        frame = checkerboard(TRICOLOR_12BIT, width=64, height=64, bit_depth=12)

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
        from display_patterns.image_generators import PatternGenerator

        roi = ROI(x=100, y=100, width=1720, height=880)
        colors = [[2000, 2000, 2000], [0, 0, 0]]
        legacy = PatternGenerator(
            bit_depth=12, width=1920, height=1080, roi=roi
        ).generate(colors)
        pure = checkerboard(colors, width=1920, height=1080, bit_depth=12, roi=roi)

        assert pure.dtype == legacy.dtype
        np.testing.assert_array_equal(pure, legacy)

    def test_explicit_numpy_namespace_matches_default(self) -> None:
        """``xp`` defaults to numpy; passing numpy explicitly is identical."""
        default = checkerboard(TRICOLOR_12BIT, width=16, height=16, bit_depth=12)
        explicit = checkerboard(
            TRICOLOR_12BIT, width=16, height=16, bit_depth=12, xp=np
        )

        np.testing.assert_array_equal(default, explicit)

    def test_dtype_override(self) -> None:
        """The caller may override the integer dtype of the result."""
        frame = checkerboard(
            TRICOLOR_12BIT, width=8, height=8, bit_depth=12, dtype=np.uint32
        )

        assert frame.dtype == np.uint32
        assert set(np.unique(frame)) == {0, 2048, 4095}


def test_checkerboard_renders_identically_under_torch() -> None:
    """The same parameters render the same values through torch's
    namespace (§req:success-criteria backends). Skips where torch is
    absent — the numpy leg above exercises the identical code path."""
    torch = pytest.importorskip("torch")

    expected = checkerboard(TRICOLOR_12BIT, width=32, height=32, bit_depth=12)
    tensor = checkerboard(TRICOLOR_12BIT, width=32, height=32, bit_depth=12, xp=torch)

    np.testing.assert_array_equal(np.asarray(tensor), expected)
