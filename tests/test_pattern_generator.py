"""
Legacy class-surface tests (§spec:extraction compat).

`PatternGenerator.generate` is a six-line delegation to
:func:`display_patterns.patterns.checkerboard`; behavioral coverage
lives with the pure entry points (``test_frame_indexed_api``), and
array equality between the two surfaces is asserted there. Here we keep
only the pieces unique to the legacy surface: the ``ROI`` value type
and the ``ColorRangeError`` message contract.
"""

import pytest

from display_patterns.image_generators.checkerboard import ROI, ColorRangeError


class TestROI:
    """ROI value semantics."""

    def test_fields_and_derived_bounds(self) -> None:
        roi = ROI(x=100, y=100, width=800, height=600)
        assert (roi.x, roi.y, roi.width, roi.height) == (100, 100, 800, 600)
        assert roi.x2 == 900
        assert roi.y2 == 700

    def test_default_values(self) -> None:
        roi = ROI()
        assert (roi.x, roi.y, roi.width, roi.height) == (0, 0, 100, 100)


class TestColorRangeError:
    """The error message contract the CLI consumer matches on."""

    def test_error_message(self) -> None:
        with pytest.raises(ColorRangeError, match="bit-depth range"):
            raise ColorRangeError()

    def test_error_with_detail(self) -> None:
        error = ColorRangeError("Bit depth: 12")
        assert error.__notes__ == ["Bit depth: 12"]
